/*!
 * \file nll_loss.cpp
 * \brief NllLoss kernel 入口，schMode 分派见 docs/AI_PROJECT.md §4.4
 */

#include "nll_loss.h"

template <uint32_t schMode>
__global__ __aicore__ void nll_loss(
    GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR total_weight, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NllLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(NllLossTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(NLLLOSS_TPL_SCH_MODE_0)) {
        NsNllLoss::NllLoss<half, int32_t> op;
        op.Init(x, target, weight, y, total_weight, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(NLLLOSS_TPL_SCH_MODE_1)) {
        NsNllLoss::NllLoss<half, int64_t> op;
        op.Init(x, target, weight, y, total_weight, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(NLLLOSS_TPL_SCH_MODE_2)) {
        NsNllLoss::NllLoss<bfloat16_t, int32_t> op;
        op.Init(x, target, weight, y, total_weight, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(NLLLOSS_TPL_SCH_MODE_3)) {
        NsNllLoss::NllLoss<bfloat16_t, int64_t> op;
        op.Init(x, target, weight, y, total_weight, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(NLLLOSS_TPL_SCH_MODE_4)) {
        NsNllLoss::NllLoss<float, int32_t> op;
        op.Init(x, target, weight, y, total_weight, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(NLLLOSS_TPL_SCH_MODE_5)) {
        NsNllLoss::NllLoss<float, int64_t> op;
        op.Init(x, target, weight, y, total_weight, workspace, &tilingData);
        op.Process();
    }
}
