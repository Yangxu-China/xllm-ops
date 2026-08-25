/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;

static ge::graphStatus InferShapeXAttentionV2(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("XAttentionV2", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter XAttentionV2 InferShape impl.");
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    gert::Shape *outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *queryShape;
    OP_LOGD(context->GetNodeName(), "XAttentionV2 InferShape end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeXAttentionV2(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("XAttentionV2", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const auto *queryDtype = context->GetInputDesc(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDtype);
    context->SetOutputDataType(0, queryDtype->GetDataType());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(XAttentionV2)
    .InferShape(InferShapeXAttentionV2)
    .InferDataType(InferDataTypeXAttentionV2);
} // namespace ops
