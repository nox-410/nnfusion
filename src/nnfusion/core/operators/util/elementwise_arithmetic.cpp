//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// Microsoft (c) 2019, NNFusion Team

#include "elementwise_arithmetic.hpp"
#include "nnfusion/core/graph/gnode.hpp"

#include "nnfusion/common/partial_shape.hpp"
#include "nnfusion/common/type/element_type.hpp"

using namespace nnfusion::op;

ElementwiseArithmetic::ElementwiseArithmetic(const std::string& node_type)
    : Op(node_type)
{
}

void ElementwiseArithmetic::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto args_et_pshape = validate_and_infer_elementwise_args(gnode);

    nnfusion::element::Type& args_et = std::get<0>(args_et_pshape);
    nnfusion::PartialShape& args_pshape = std::get<1>(args_et_pshape);

    OP_VALIDATION(this, args_et.is_dynamic() || args_et != nnfusion::element::boolean)
        << "Arguments cannot have boolean element type (argument element type: " << args_et << ").";

    gnode->set_output_type_and_shape(0, args_et, args_pshape);
}

void ElementwiseArithmetic::infer_shared_memory(std::shared_ptr<graph::GNode> gnode)
{
    auto& input_shape = gnode->get_input_shape(0);
    auto& output_shape = gnode->get_output_shape(0);
    if (input_shape.size() == output_shape.size())
    {
        m_shared_memory.clear();
        for (size_t i = 0; i < output_shape.size(); i++)
        {
            m_shared_memory.push_back(1);
        }
    }
}

std::vector<std::vector<size_t>> ElementwiseArithmetic::infer_runtime_share_memory(
    std::shared_ptr<graph::GNode> gnode, std::vector<std::vector<size_t>> inputs)
{
    auto in_reduce_vec_0 = inputs[0];
    auto in_reduce_vec_1 = inputs.size() == 2 ? inputs[1] : std::vector<size_t>(in_reduce_vec_0.size(), 1);
    auto out_shape = gnode->get_output_shape(0);
    std::vector<std::vector<size_t>> out_reduce_vec(1, std::vector<size_t>());
    
    NNFUSION_CHECK(in_reduce_vec_0.size() == in_reduce_vec_1.size()) << "Input size dims should be same for elementwise";
    for (int d = 0; d < in_reduce_vec_0.size(); ++d)
    {
        out_reduce_vec[0].push_back(std::min(in_reduce_vec_0[d] * in_reduce_vec_1[d], out_shape[d]));
    }
    return out_reduce_vec;
}
