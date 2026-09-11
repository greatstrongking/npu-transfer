/*!
 * \file nll_loss.h
 * \brief NllLoss kernel，算法见 docs/AI_PROJECT.md §5
 */

#ifndef NLLLOSS_H
#define NLLLOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "nll_loss_tiling_data.h"
#include "nll_loss_tiling_key.h"

namespace NsNllLoss {
using namespace AscendC;

constexpr int32_t BLOCK_BYTES = 32;
constexpr int32_t WS_FLOATS_PER_SLOT = 8;

template <typename T, typename TargetT>
class NllLoss {
public:
    __aicore__ inline NllLoss(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR totalWeight, GM_ADDR workspace,
        const NllLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseCoreRange();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void LoadWeight();
    __aicore__ inline void ProcessOneTile(int64_t nOffset, int64_t curN);
    __aicore__ inline void GatherNormal(int64_t curN);
    __aicore__ inline void GatherLarge(int64_t nOffset, int64_t curN);
    __aicore__ inline void ComputeTile(int64_t nOffset, int64_t curN);
    __aicore__ inline void CopyInX(int64_t gmOffset, int64_t elems);
    __aicore__ inline void CopyInTarget(int64_t gmOffset, int64_t elems);
    __aicore__ inline void CopyOutY(int64_t gmOffset, int64_t elems);
    __aicore__ inline void WaitMte2ToS();
    __aicore__ inline void WaitSToV();
    __aicore__ inline void WaitVToS();
    __aicore__ inline void WaitVToMte3();
    __aicore__ inline void WaitSToMte3();
    __aicore__ inline void WritePartialToWorkspace();
    __aicore__ inline void ReduceAndWriteOutputs();
    __aicore__ inline void WriteScalarOut(GlobalTensor<T>& dst, float value, LocalTensor<T>& outLocal);
    __aicore__ inline int64_t AlignElems(int64_t elems, int64_t bytesPerElem);

    TPipe pipe;
    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> targetBuf;
    TBuf<TPosition::VECCALC> weightBuf;
    TBuf<TPosition::VECCALC> validXBuf;
    TBuf<TPosition::VECCALC> validWBuf;
    TBuf<TPosition::VECCALC> fpXBuf;
    TBuf<TPosition::VECCALC> fpWBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> workBuf;

    GlobalTensor<T> xGM;
    GlobalTensor<TargetT> targetGM;
    GlobalTensor<T> weightGM;
    GlobalTensor<T> yGM;
    GlobalTensor<T> totalWeightGM;
    GlobalTensor<float> wsGM;

    const NllLossTilingData* tiling_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t nOffset_ = 0;
    int64_t nLen_ = 0;
    bool hasWeight_ = false;
    float localLoss_ = 0.0f;
    float localWeight_ = 0.0f;
};

template <typename T, typename TargetT>
__aicore__ inline int64_t NllLoss<T, TargetT>::AlignElems(int64_t elems, int64_t bytesPerElem)
{
    int64_t blockElems = BLOCK_BYTES / bytesPerElem;
    if (blockElems <= 0) {
        return elems;
    }
    return ((elems + blockElems - 1) / blockElems) * blockElems;
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::WaitMte2ToS()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::WaitSToV()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId);
    WaitFlag<HardEvent::S_V>(eventId);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::WaitVToS()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::WaitVToMte3()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::WaitSToMte3()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId);
    WaitFlag<HardEvent::S_MTE3>(eventId);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::Init(
    GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR totalWeight, GM_ADDR workspace,
    const NllLossTilingData* tilingData)
{
    tiling_ = tilingData;
    blockIdx_ = GetBlockIdx();
    hasWeight_ = (tiling_->hasWeight != 0) && (weight != nullptr);
    xGM.SetGlobalBuffer((__gm__ T*)x);
    targetGM.SetGlobalBuffer((__gm__ TargetT*)target);
    if (hasWeight_) {
        weightGM.SetGlobalBuffer((__gm__ T*)weight);
    }
    yGM.SetGlobalBuffer((__gm__ T*)y);
    totalWeightGM.SetGlobalBuffer((__gm__ T*)totalWeight);
    wsGM.SetGlobalBuffer((__gm__ float*)workspace);
    ParseCoreRange();
    InitBuffers();
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::ParseCoreRange()
{
    const bool isLast = (blockIdx_ == tiling_->needCoreNum - 1);
    nLen_ = isLast ? tiling_->lastCoreSize : tiling_->perCoreSize;
    nOffset_ = blockIdx_ * tiling_->perCoreSize;
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::InitBuffers()
{
    int64_t xBytes = tiling_->xUbElems * static_cast<int64_t>(sizeof(T));
    int64_t tBytes = tiling_->targetUbElems * static_cast<int64_t>(sizeof(TargetT));
    int64_t wBytes = tiling_->weightUbElems * static_cast<int64_t>(sizeof(T));
    int64_t vBytes = tiling_->targetUbElems * static_cast<int64_t>(sizeof(T));
    int64_t fpBytes = tiling_->targetUbElems * static_cast<int64_t>(sizeof(float));
    if (xBytes < BLOCK_BYTES) {
        xBytes = BLOCK_BYTES;
    }
    if (tBytes < BLOCK_BYTES) {
        tBytes = BLOCK_BYTES;
    }
    if (wBytes < BLOCK_BYTES) {
        wBytes = BLOCK_BYTES;
    }
    if (vBytes < BLOCK_BYTES) {
        vBytes = BLOCK_BYTES;
    }
    if (fpBytes < BLOCK_BYTES) {
        fpBytes = BLOCK_BYTES;
    }
    pipe.InitBuffer(xBuf, xBytes);
    pipe.InitBuffer(targetBuf, tBytes);
    pipe.InitBuffer(weightBuf, wBytes);
    pipe.InitBuffer(validXBuf, vBytes);
    pipe.InitBuffer(validWBuf, vBytes);
    pipe.InitBuffer(fpXBuf, fpBytes);
    pipe.InitBuffer(fpWBuf, fpBytes);
    pipe.InitBuffer(tmpBuf, fpBytes);
    pipe.InitBuffer(workBuf, fpBytes);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::CopyInX(int64_t gmOffset, int64_t elems)
{
    LocalTensor<T> xLocal = xBuf.Get<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(elems * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGM[gmOffset], params, padParams);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::CopyInTarget(int64_t gmOffset, int64_t elems)
{
    LocalTensor<TargetT> tLocal = targetBuf.Get<TargetT>();
    DataCopyExtParams params{1, static_cast<uint32_t>(elems * sizeof(TargetT)), 0, 0, 0};
    DataCopyPadExtParams<TargetT> padParams{false, 0, 0, 0};
    DataCopyPad(tLocal, targetGM[gmOffset], params, padParams);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::CopyOutY(int64_t gmOffset, int64_t elems)
{
    LocalTensor<T> yLocal = validXBuf.Get<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(elems * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGM[gmOffset], yLocal, params);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::LoadWeight()
{
    if (tiling_->tilingMode != NLLLOSS_TILING_MODE_NORMAL || !hasWeight_) {
        return;
    }
    LocalTensor<T> wLocal = weightBuf.Get<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(tiling_->cSize * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(wLocal, weightGM[0], params, padParams);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::GatherNormal(int64_t curN)
{
    LocalTensor<T> xLocal = xBuf.Get<T>();
    LocalTensor<TargetT> tLocal = targetBuf.Get<TargetT>();
    LocalTensor<T> wLocal = weightBuf.Get<T>();
    LocalTensor<T> vx = validXBuf.Get<T>();
    LocalTensor<T> vw = validWBuf.Get<T>();
    int64_t alignN = AlignElems(curN, static_cast<int64_t>(sizeof(T)));
    Duplicate(vx, static_cast<T>(0), alignN);
    Duplicate(vw, static_cast<T>(0), alignN);
    WaitVToS();
    WaitMte2ToS();
    for (int64_t i = 0; i < curN; ++i) {
        int64_t cls = static_cast<int64_t>(tLocal.GetValue(i));
        if (cls == tiling_->ignoreIndex) {
            vx.SetValue(i, static_cast<T>(0));
            vw.SetValue(i, static_cast<T>(0));
        } else {
            vx.SetValue(i, xLocal.GetValue(i * tiling_->cSize + cls));
            if (hasWeight_) {
                vw.SetValue(i, wLocal.GetValue(cls));
            } else {
                vw.SetValue(i, static_cast<T>(1));
            }
        }
    }
    WaitSToV();
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::GatherLarge(int64_t nOffset, int64_t curN)
{
    LocalTensor<TargetT> tLocal = targetBuf.Get<TargetT>();
    LocalTensor<T> xLocal = xBuf.Get<T>();
    LocalTensor<T> wLocal = weightBuf.Get<T>();
    LocalTensor<T> vx = validXBuf.Get<T>();
    LocalTensor<T> vw = validWBuf.Get<T>();
    int64_t alignN = AlignElems(curN, static_cast<int64_t>(sizeof(T)));
    Duplicate(vx, static_cast<T>(0), alignN);
    Duplicate(vw, static_cast<T>(0), alignN);
    WaitVToS();
    WaitMte2ToS();
    DataCopyExtParams oneX{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    for (int64_t i = 0; i < curN; ++i) {
        int64_t cls = static_cast<int64_t>(tLocal.GetValue(i));
        if (cls == tiling_->ignoreIndex) {
            vx.SetValue(i, static_cast<T>(0));
            vw.SetValue(i, static_cast<T>(0));
            continue;
        }
        int64_t xOffset = (nOffset + i) * tiling_->cSize + cls;
        DataCopyPad(xLocal, xGM[xOffset], oneX, padParams);
        if (hasWeight_) {
            DataCopyPad(wLocal, weightGM[cls], oneX, padParams);
        }
        WaitMte2ToS();
        vx.SetValue(i, xLocal.GetValue(0));
        if (hasWeight_) {
            vw.SetValue(i, wLocal.GetValue(0));
        } else {
            vw.SetValue(i, static_cast<T>(1));
        }
    }
    WaitSToV();
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::ComputeTile(int64_t nOffset, int64_t curN)
{
    LocalTensor<T> vx = validXBuf.Get<T>();
    LocalTensor<T> vw = validWBuf.Get<T>();
    LocalTensor<float> fpX = fpXBuf.Get<float>();
    LocalTensor<float> fpW = fpWBuf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    LocalTensor<float> work = workBuf.Get<float>();
    int64_t calcN = AlignElems(curN, static_cast<int64_t>(sizeof(float)));

    Duplicate(fpX, 0.0f, calcN);
    Duplicate(fpW, 0.0f, calcN);
    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(fpX, vx, static_cast<float>(-1), curN);
        Mul(fpX, fpX, vw, curN);
        Muls(fpW, vw, static_cast<float>(1), curN);
    } else {
        Cast(fpX, vx, RoundMode::CAST_NONE, curN);
        Cast(fpW, vw, RoundMode::CAST_NONE, curN);
        Muls(fpX, fpX, static_cast<float>(-1), curN);
        Mul(fpX, fpX, fpW, curN);
    }

    ReduceSum(tmp, fpX, work, curN);
    WaitVToS();
    localLoss_ += tmp.GetValue(0);

    WaitSToV();
    ReduceSum(tmp, fpW, work, curN);
    WaitVToS();
    localWeight_ += tmp.GetValue(0);

    if (tiling_->reduction == NLLLOSS_REDUCTION_NONE) {
        if constexpr (sizeof(T) == sizeof(float)) {
            WaitVToMte3();
            DataCopyExtParams params{1, static_cast<uint32_t>(curN * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGM[nOffset], fpX, params);
        } else {
            WaitSToV();
            Cast(vx, fpX, RoundMode::CAST_RINT, curN);
            WaitVToMte3();
            CopyOutY(nOffset, curN);
        }
    }
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::ProcessOneTile(int64_t nOffset, int64_t curN)
{
    if (curN <= 0) {
        return;
    }
    CopyInTarget(nOffset, curN);
    if (tiling_->tilingMode == NLLLOSS_TILING_MODE_NORMAL) {
        CopyInX(nOffset * tiling_->cSize, curN * tiling_->cSize);
        GatherNormal(curN);
    } else {
        GatherLarge(nOffset, curN);
    }
    ComputeTile(nOffset, curN);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::WritePartialToWorkspace()
{
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    Duplicate(tmp, 0.0f, WS_FLOATS_PER_SLOT);
    WaitVToS();
    tmp.SetValue(0, localLoss_);
    WaitSToMte3();
    DataCopyExtParams params{1, static_cast<uint32_t>(BLOCK_BYTES), 0, 0, 0};
    DataCopyPad(wsGM[blockIdx_ * WS_FLOATS_PER_SLOT], tmp, params);

    Duplicate(tmp, 0.0f, WS_FLOATS_PER_SLOT);
    WaitVToS();
    tmp.SetValue(0, localWeight_);
    WaitSToMte3();
    int64_t wBase = tiling_->needCoreNum * WS_FLOATS_PER_SLOT;
    DataCopyPad(wsGM[wBase + blockIdx_ * WS_FLOATS_PER_SLOT], tmp, params);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::WriteScalarOut(
    GlobalTensor<T>& dst, float value, LocalTensor<T>& outLocal)
{
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    Duplicate(tmp, 0.0f, WS_FLOATS_PER_SLOT);
    WaitVToS();
    tmp.SetValue(0, value);
    if constexpr (sizeof(T) == sizeof(float)) {
        outLocal.SetValue(0, static_cast<T>(value));
        WaitSToMte3();
    } else {
        WaitSToV();
        Cast(outLocal, tmp, RoundMode::CAST_RINT, WS_FLOATS_PER_SLOT);
        WaitVToMte3();
    }
    DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(dst[0], outLocal, outParams);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::ReduceAndWriteOutputs()
{
    if (blockIdx_ != 0) {
        return;
    }
    LocalTensor<float> fpX = fpXBuf.Get<float>();
    float sumLoss = 0.0f;
    float sumW = 0.0f;
    DataCopyExtParams params{1, static_cast<uint32_t>(BLOCK_BYTES), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    int64_t wBase = tiling_->needCoreNum * WS_FLOATS_PER_SLOT;
    for (int64_t i = 0; i < tiling_->needCoreNum; ++i) {
        DataCopyPad(fpX, wsGM[i * WS_FLOATS_PER_SLOT], params, padParams);
        WaitMte2ToS();
        sumLoss += fpX.GetValue(0);
        DataCopyPad(fpX, wsGM[wBase + i * WS_FLOATS_PER_SLOT], params, padParams);
        WaitMte2ToS();
        sumW += fpX.GetValue(0);
    }
    float yVal = sumLoss;
    if (tiling_->reduction == NLLLOSS_REDUCTION_MEAN) {
        yVal = (sumW == 0.0f) ? 0.0f : (sumLoss / sumW);
    }
    LocalTensor<T> yLocal = validXBuf.Get<T>();
    if (tiling_->reduction != NLLLOSS_REDUCTION_NONE) {
        WriteScalarOut(yGM, yVal, yLocal);
    }
    LocalTensor<T> wOut = validWBuf.Get<T>();
    WriteScalarOut(totalWeightGM, sumW, wOut);
}

template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::Process()
{
    if (blockIdx_ >= tiling_->needCoreNum) {
        SyncAll();
        return;
    }
    LoadWeight();
    localLoss_ = 0.0f;
    localWeight_ = 0.0f;
    int64_t remain = nLen_;
    int64_t cursor = nOffset_;
    int64_t tileLen = tiling_->tileN;
    if (tileLen < 1) {
        tileLen = 1;
    }
    while (remain > 0) {
        int64_t curN = (remain < tileLen) ? remain : tileLen;
        ProcessOneTile(cursor, curN);
        cursor += curN;
        remain -= curN;
    }
    WritePartialToWorkspace();
    SyncAll();
    ReduceAndWriteOutputs();
}

} // namespace NsNllLoss
#endif // NLLLOSS_H
