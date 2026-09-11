/*!
 * \file nll_loss_tiling_data.h
 * \brief Host/Kernel 共享 TilingData，字段顺序见 docs/AI_PROJECT.md §4.1
 */

#ifndef _NLLLOSS_TILING_DATA_H_
#define _NLLLOSS_TILING_DATA_H_

#include <cstdint>

constexpr int64_t NLLLOSS_TILING_MODE_NORMAL = 1;
constexpr int64_t NLLLOSS_TILING_MODE_LARGE = 2;
constexpr int64_t NLLLOSS_REDUCTION_NONE = 0;
constexpr int64_t NLLLOSS_REDUCTION_MEAN = 1;
constexpr int64_t NLLLOSS_REDUCTION_SUM = 2;
constexpr int64_t NLLLOSS_WS_SLOT_BYTES = 32;

struct NllLossTilingData {
    int64_t tilingMode = 1;
    int64_t needCoreNum = 1;
    int64_t nSize = 0;
    int64_t cSize = 0;
    int64_t perCoreSize = 1;
    int64_t perCoreLoopCount = 0;
    int64_t perCoreLeftSize = 0;
    int64_t lastCoreSize = 0;
    int64_t lastCoreLoopCount = 0;
    int64_t lastCoreLeftSize = 0;
    int64_t xUbElems = 0;
    int64_t targetUbElems = 0;
    int64_t weightUbElems = 0;
    int64_t ignoreIndex = -100;
    int64_t reduction = 1;
    int64_t hasWeight = 0;
    int64_t tileN = 1;
    int64_t xDimNum = 2;
};

#endif
