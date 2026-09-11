#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "nll_loss_tiling_data.h"

namespace NllLossUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "NllLoss";

struct NllLossTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> targetShape;
    ge::DataType targetDtype;
    ge::Format targetFormat;
    std::initializer_list<int64_t> weightShape;
    ge::DataType weightDtype;
    ge::Format weightFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::initializer_list<int64_t> total_weightShape;
    ge::DataType total_weightDtype;
    ge::Format total_weightFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/nll_loss_tiling_key.h 和 op_host/nll_loss_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/nll_loss_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/nll_loss_tiling.cpp 中 GetWorkspaceSize 的逻辑
static NllLossTestParam testCases[] = {
    {"nll_loss_0", {3, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {3}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, EMPTY_EXPECT_TILING_DATA, {192}, 64, 262144, 4096},
};

class NllLossTilingTest : public testing::TestWithParam<NllLossTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "NllLossTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "NllLossTilingTest TearDown." << std::endl;
    }
};

struct NllLossCompileInfo {} compileInfo;

static void TestOneParamCase(const NllLossTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape targetShape = {param.targetShape, param.targetShape};
    gert::StorageShape weightShape = {param.weightShape, param.weightShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape total_weightShape = {param.total_weightShape, param.total_weightShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
        {targetShape, param.targetDtype, param.targetFormat},
        {weightShape, param.weightDtype, param.weightFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat},
        {total_weightShape, param.total_weightDtype, param.total_weightFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("reduction", Ops::Math::AnyValue::CreateFrom<std::string>("mean")));
    attrs_.push_back(gert::TilingContextPara::OpAttr("ignore_index", Ops::Math::AnyValue::CreateFrom<int64_t>(-100)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(NllLossTilingTest, tiling_test)
{
    const NllLossTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    NllLossTilingTests,
    NllLossTilingTest,
    testing::ValuesIn(testCases));

}
