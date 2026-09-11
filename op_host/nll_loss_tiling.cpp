/*!
 * \file nll_loss_tiling.cpp
 * \brief NllLoss Tiling，算法见 docs/AI_PROJECT.md §4
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/nll_loss_tiling_data.h"
#include "../op_kernel/nll_loss_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

namespace {
constexpr int64_t UB_RESERVE = 8192;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t MIN_TILE_N = 1;

static int64_t AlignUp(int64_t value, int64_t align)
{
    if (align <= 0) {
        return value;
    }
    return (value + align - 1) / align * align;
}

static int64_t GetDtypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_INT64:
            return 8;
        default:
            return 4;
    }
}

static int64_t ParseReduction(const char* reduction)
{
    if (reduction == nullptr) {
        return NLLLOSS_REDUCTION_MEAN;
    }
    if (std::strcmp(reduction, "none") == 0) {
        return NLLLOSS_REDUCTION_NONE;
    }
    if (std::strcmp(reduction, "sum") == 0) {
        return NLLLOSS_REDUCTION_SUM;
    }
    return NLLLOSS_REDUCTION_MEAN;
}

static uint64_t GetSchMode(ge::DataType xDtype, ge::DataType targetDtype)
{
    const bool isInt64 = (targetDtype == ge::DT_INT64);
    if (xDtype == ge::DT_FLOAT16) {
        return isInt64 ? NLLLOSS_TPL_SCH_MODE_1 : NLLLOSS_TPL_SCH_MODE_0;
    }
    if (xDtype == ge::DT_BF16) {
        return isInt64 ? NLLLOSS_TPL_SCH_MODE_3 : NLLLOSS_TPL_SCH_MODE_2;
    }
    return isInt64 ? NLLLOSS_TPL_SCH_MODE_5 : NLLLOSS_TPL_SCH_MODE_4;
}

static void SplitLoops(int64_t coreSize, int64_t tileN, int64_t& loopCount, int64_t& leftSize, int64_t& realTileN)
{
    realTileN = tileN;
    if (coreSize <= 0) {
        loopCount = 0;
        leftSize = 0;
        return;
    }
    if (realTileN > coreSize) {
        realTileN = coreSize;
    }
    if (realTileN < MIN_TILE_N) {
        realTileN = MIN_TILE_N;
    }
    loopCount = coreSize / realTileN;
    leftSize = coreSize % realTileN;
    if (loopCount == 0) {
        loopCount = 1;
        leftSize = 0;
        realTileN = coreSize;
    }
}
} // namespace

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NllLossTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t aivNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, aivNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* xStorage = context->GetInputShape(0);
    const gert::StorageShape* targetStorage = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetStorage);
    const gert::Shape& xShape = xStorage->GetStorageShape();

    const auto xDesc = context->GetInputDesc(0);
    const auto targetDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);

    int64_t xDimNum = static_cast<int64_t>(xShape.GetDimNum());
    int64_t cSize = 1;
    int64_t nSize = 1;
    if (xDimNum <= 0) {
        cSize = 1;
        nSize = 1;
    } else if (xDimNum == 1) {
        cSize = xShape.GetDim(0);
        nSize = 1;
    } else {
        cSize = xShape.GetDim(xDimNum - 1);
        nSize = 1;
        for (int64_t i = 0; i < xDimNum - 1; ++i) {
            nSize *= xShape.GetDim(i);
        }
    }
    if (cSize <= 0) {
        cSize = 1;
    }
    if (nSize < 0) {
        nSize = 0;
    }

    const gert::StorageShape* weightStorage = context->GetOptionalInputShape(2);
    const int64_t hasWeight = (weightStorage != nullptr) ? 1 : 0;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t reduction = ParseReduction(attrs->GetStr(0));
    const int64_t* ignorePtr = attrs->GetInt(1);
    const int64_t ignoreIndex = (ignorePtr == nullptr) ? -100 : *ignorePtr;

    const int64_t xBytes = GetDtypeBytes(xDesc->GetDataType());
    const int64_t tBytes = GetDtypeBytes(targetDesc->GetDataType());
    const int64_t blockElemsX = BLOCK_BYTES / xBytes;

    int64_t usedCores = 1;
    int64_t perCoreSize = 0;
    int64_t lastCoreSize = 0;
    if (nSize == 0) {
        usedCores = 1;
        perCoreSize = 0;
        lastCoreSize = 0;
    } else {
        usedCores = std::min(nSize, aivNum);
        perCoreSize = CeilDiv(nSize, usedCores);
        usedCores = CeilDiv(nSize, perCoreSize);
        lastCoreSize = nSize - perCoreSize * (usedCores - 1);
    }

    const int64_t availUb = std::max(static_cast<int64_t>(ubSize) - UB_RESERVE, static_cast<int64_t>(BLOCK_BYTES * 10));
    // 预留 ReduceSum workBuf(256B) 以及 Cast/Duplicate 最小 64 元素的 fp/valid buffer
    const int64_t bufOverhead = BLOCK_BYTES * 10 + 2048;
    int64_t weightElems = AlignUp(std::max(cSize, blockElemsX), blockElemsX);
    int64_t weightBytes = weightElems * xBytes;
    const int64_t bytesPerNNormal = cSize * xBytes + tBytes + 2 * xBytes + 16;
    const int64_t bytesPerNLarge = tBytes + 2 * xBytes + 16;
    int64_t tilingMode = NLLLOSS_TILING_MODE_NORMAL;
    int64_t maxTileN = MIN_TILE_N;
    const int64_t normalNeed = cSize * xBytes + weightBytes + bufOverhead;
    if (normalNeed > availUb || bytesPerNNormal <= 0) {
        tilingMode = NLLLOSS_TILING_MODE_LARGE;
        weightElems = blockElemsX;
        weightBytes = weightElems * xBytes;
        maxTileN = std::max((availUb - bufOverhead - weightBytes) / std::max(bytesPerNLarge, static_cast<int64_t>(1)),
                            MIN_TILE_N);
    } else {
        maxTileN = (availUb - weightBytes - bufOverhead) / bytesPerNNormal;
        if (maxTileN < MIN_TILE_N) {
            tilingMode = NLLLOSS_TILING_MODE_LARGE;
            weightElems = blockElemsX;
            weightBytes = weightElems * xBytes;
            maxTileN = std::max(
                (availUb - bufOverhead - weightBytes) / std::max(bytesPerNLarge, static_cast<int64_t>(1)), MIN_TILE_N);
        }
    }

    int64_t tileN = maxTileN;
    int64_t perLoop = 0;
    int64_t perLeft = 0;
    int64_t lastLoop = 0;
    int64_t lastLeft = 0;
    int64_t realTileFront = tileN;
    int64_t realTileLast = tileN;
    SplitLoops(perCoreSize, tileN, perLoop, perLeft, realTileFront);
    SplitLoops(lastCoreSize, tileN, lastLoop, lastLeft, realTileLast);
    const int64_t usedTileN = std::max(realTileFront, realTileLast);
    const int64_t targetUbElems = AlignUp(std::max(usedTileN, MIN_TILE_N), BLOCK_BYTES / std::max(tBytes, static_cast<int64_t>(1)));
    const int64_t xUbElems = (tilingMode == NLLLOSS_TILING_MODE_NORMAL) ?
        AlignUp(usedTileN * cSize, blockElemsX) :
        blockElemsX;

    NllLossTilingData* tiling = context->GetTilingData<NllLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->tilingMode = tilingMode;
    tiling->needCoreNum = usedCores;
    tiling->nSize = nSize;
    tiling->cSize = cSize;
    tiling->perCoreSize = perCoreSize;
    tiling->perCoreLoopCount = perLoop;
    tiling->perCoreLeftSize = perLeft;
    tiling->lastCoreSize = lastCoreSize;
    tiling->lastCoreLoopCount = lastLoop;
    tiling->lastCoreLeftSize = lastLeft;
    tiling->xUbElems = xUbElems;
    tiling->targetUbElems = targetUbElems;
    tiling->weightUbElems = weightElems;
    tiling->ignoreIndex = ignoreIndex;
    tiling->reduction = reduction;
    tiling->hasWeight = hasWeight;
    tiling->tileN = usedTileN;
    tiling->xDimNum = std::max(xDimNum, static_cast<int64_t>(1));

    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = static_cast<size_t>(usedCores * NLLLOSS_WS_SLOT_BYTES * 2);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(GetSchMode(xDesc->GetDataType(), targetDesc->GetDataType()));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForNllLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct NllLossCompileInfo {};

IMPL_OP_OPTILING(NllLoss).Tiling(NllLossTilingFunc).TilingParse<NllLossCompileInfo>(TilingParseForNllLoss);

} // namespace optiling
