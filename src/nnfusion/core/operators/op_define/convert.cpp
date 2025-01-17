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

#include "convert.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace nnfusion::op;

Convert::Convert(const nnfusion::element::Type& element_type)
    : Op("Convert")
    , m_element_type(element_type)
{
}

void Convert::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    gnode->set_output_type_and_shape(0, m_element_type, gnode->get_input_shape(0));
}

std::vector<std::vector<size_t>> Convert::infer_runtime_share_memory(std::shared_ptr<graph::GNode> gnode,
    std::vector<std::vector<size_t>> in_reduce_vecs)
{
    return in_reduce_vecs;
}
