/*!
 * \file test_nll_loss.cpp
 * \brief NllLoss 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "nll_loss_tiling.h"
#include "../../../op_kernel/nll_loss.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class NllLossKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "NllLossKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "NllLossKernelTest TearDown" << endl;
    }
};

TEST_F(NllLossKernelTest, test_kernel_run)
{
    constexpr size_t size = 15;
    constexpr size_t tilingDataSize = sizeof(NllLossTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 15 * 2;
    constexpr size_t targetByteSize = 3 * 4;
    constexpr size_t weightByteSize = 5 * 2;
    constexpr size_t yByteSize = 1 * 2;
    constexpr size_t total_weightByteSize = 1 * 2;
    std::vector<float> xHost(15, 1);
    std::vector<int32_t> targetHost(3, 1);
    std::vector<float> weightHost(5, 1);
    std::vector<float> yHost(1, 0);
    std::vector<float> total_weightHost(1, 0);
    
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* total_weight = (uint8_t*)AscendC::GmAlloc(total_weightByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(256);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    for (size_t _i = 0; _i < 15; _i++) { uint16_t _h = FloatToHalf(xHost[_i]); memcpy(x + _i * 2, &_h, 2); }
    memcpy(target, targetHost.data(), targetByteSize);
    for (size_t _i = 0; _i < 5; _i++) { uint16_t _h = FloatToHalf(weightHost[_i]); memcpy(weight + _i * 2, &_h, 2); }
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/nll_loss_tiling_data.h 中的字段定义
    //         参考 op_host/nll_loss_tiling.cpp 中的 tiling 计算逻辑
    NllLossTilingData* tilingData = reinterpret_cast<NllLossTilingData*>(tiling);
    tilingData->tilingMode = NLLLOSS_TILING_MODE_NORMAL;
    tilingData->needCoreNum = 1;
    tilingData->nSize = 3;
    tilingData->cSize = 5;
    tilingData->perCoreSize = 3;
    tilingData->perCoreLoopCount = 1;
    tilingData->perCoreLeftSize = 0;
    tilingData->lastCoreSize = 3;
    tilingData->lastCoreLoopCount = 1;
    tilingData->lastCoreLeftSize = 0;
    tilingData->xUbElems = 16;
    tilingData->targetUbElems = 8;
    tilingData->weightUbElems = 16;
    tilingData->ignoreIndex = -100;
    tilingData->reduction = NLLLOSS_REDUCTION_MEAN;
    tilingData->hasWeight = 1;
    tilingData->tileN = 3;
    tilingData->xDimNum = 2;
    
    // TODO: tilingKey 应与 op_host/nll_loss_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((nll_loss<0>), numBlocks, x, target, weight, y, total_weight, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs("float16_output_nll_loss_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    memcpy(total_weightHost.data(), total_weight, total_weightByteSize);
    { std::ofstream _ofs("float16_output_nll_loss_1.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(total_weightHost.data()), total_weightByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(y);
    AscendC::GmFree(total_weight);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
