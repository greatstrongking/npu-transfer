/*!
 * \file nll_loss_infershape.cpp
 * \brief NllLoss InferShape / InferDataType，规则见 docs/AI_PROJECT.md §2.3
 */

#include <cstring>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

namespace ops {

static bool IsNoneReduction(const gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const char* reduction = attrs->GetStr(0);
    return reduction != nullptr && std::strcmp(reduction, "none") == 0;
}

static ge::graphStatus InferShapeNllLoss(gert::InferShapeContext* context)
{
    const gert::Shape* targetShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    gert::Shape* totalWeightShape = context->GetOutputShape(1);
    if (yShape == nullptr || totalWeightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    totalWeightShape->SetDimNum(1);
    totalWeightShape->SetDim(0, 1);

    if (IsNoneReduction(context) && targetShape != nullptr) {
        if (targetShape->GetDimNum() == 0) {
            yShape->SetDimNum(1);
            yShape->SetDim(0, 1);
        } else {
            *yShape = *targetShape;
        }
    } else {
        yShape->SetDimNum(1);
        yShape->SetDim(0, 1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNllLoss(gert::InferDataTypeContext* context)
{
    ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    context->SetOutputDataType(1, xDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NllLoss).InferShape(InferShapeNllLoss).InferDataType(InferDataTypeNllLoss);

} // namespace ops
