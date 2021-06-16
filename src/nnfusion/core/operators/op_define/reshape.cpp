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

#include <algorithm>
#include <iostream>

#include "nnfusion/core/graph/gnode.hpp"
#include "reshape.hpp"

using namespace std;
using namespace nnfusion::op;

Reshape::Reshape(const nnfusion::AxisVector& input_order, const nnfusion::Shape& output_shape)
    : Op("Reshape")
    , m_input_order(input_order)
    , m_output_shape(output_shape)
{
}

void Reshape::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto& input_shape = gnode->get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    // Check that the input axis order is a permutation of (0,...,n-1) for some n.
    for (size_t i = 0; i < m_input_order.size(); i++)
    {
        OP_VALIDATION(this, find(begin(m_input_order), end(m_input_order), i) != end(m_input_order))
            << "Input axis order is not a permutation of argument's axis indices (axis order: "
            << m_input_order << ", argument shape: " << input_shape << ").";
    }

    // TODO(amprocte): should be possible to move around unknown dims in the input shape.
    if (input_rank.is_static())
    {
        OP_VALIDATION(this, m_input_order.size() == size_t(input_rank))
            << "Input axis order is not a permutation of argument's axis indices (axis order: "
            << m_input_order << ", argument shape: " << input_shape << ").";

        for (size_t i = 0; i < size_t(input_rank); i++)
        {
            auto it = find(begin(m_input_order), end(m_input_order), i);
            OP_VALIDATION(this, it != end(m_input_order))
                << "Input axis order is not a permutation of argument's axis indices (axis order: "
                << m_input_order << ", argument shape: " << input_shape << ").";
        }

        // TODO(amprocte): make a partial_shape_size() analogous to shape_size().
        nnfusion::Dimension input_shape_product = 1;
        for (size_t i = 0; i < size_t(input_rank); i++)
        {
            input_shape_product *= input_shape[i];
        }

        if (input_shape_product.is_static())
        {
            OP_VALIDATION(this, size_t(input_shape_product) == nnfusion::shape_size(m_output_shape))
                << "Product of output shape dimensions does not match product of argument shape "
                   "dimensions "
                << "(output shape: " << m_output_shape << ", argument shape: " << input_shape
                << ").";
        }
    }

    if (!std::is_sorted(m_input_order.begin(), m_input_order.end()))
    {
        m_is_transpose = true;
    }

    if (m_is_transpose)
    {
        // Data layout will change if:
        //   1. m_is_transpose == true, and
        //   2. rank order changed except for the rank whose dim == 1.
        // For example:
        //   1. [1024, 200, 1, 50] => [1024, 1, 200, 50], m_input_order = [0, 2, 1, 3]
        //      ignore the rank whose dim == 1, the order = [2], layout needn't change.
        //
        //   2. [1024, 200, 1, 50] => [1024, 50, 1, 200], m_input_order = [0, 3, 2, 1]
        //      ignore the rank whose dim == 1, the order = [3, 1], layout need change.
        size_t output_size = shape_size(m_output_shape);
        size_t begin = 0, end = 0;
        bool find_begin = false;
        for (size_t i = 0; i < m_input_order.size(); ++i)
        {
            if (m_input_order[i] != i)
            {
                if (find_begin == false)
                {
                    begin = i;
                    find_begin = true;
                }
                else
                {
                    end = i;
                }
            }
        }
        nnfusion::AxisVector order;
        for (size_t i = begin; i <= end; ++i)
        {
            if (size_t(input_shape[m_input_order[i]]) != 1)
            {
                order.push_back(m_input_order[i]);
            }
        }
        if (!std::is_sorted(order.begin(), order.end()))
        {
            m_is_layout_change = true;
        }
    }
    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), m_output_shape);
}

void Reshape::infer_shared_memory(std::shared_ptr<graph::GNode> gnode)
{
    auto& input_shape = gnode->get_input_shape(0);
    auto& output_shape = gnode->get_output_shape(0);
    if (!get_is_layout_change() || shape_size(output_shape) < 2)
    {
        m_shared_memory.clear();
        for (size_t i = 0; i < output_shape.size(); i++)
            m_shared_memory.push_back(1);
    }
    else
    {
        size_t len = m_input_order.size();
        if (m_input_order[len - 1] == len - 2 && m_input_order[len - 2] == len - 1)
        {
            bool trans_inner = true;
            for (size_t i = 0; i < len - 2; i++)
            {
                if (m_input_order[i] != i)
                {
                    trans_inner = false;
                    break;
                }
            }

            if (trans_inner)
            {
                m_shared_memory.clear();
                for (size_t i = 0; i < len - 2; i++)
                {
                    m_shared_memory.push_back(1);
                }
                m_shared_memory.push_back(input_shape[len - 2]);
                m_shared_memory.push_back(input_shape[len - 1]);
            }
        }
    }
}

std::vector<std::vector<size_t>> Reshape::infer_runtime_share_memory(std::shared_ptr<graph::GNode> gnode,
                                                                     std::vector<std::vector<size_t>> in_reduce_vecs)
{
    auto op = static_pointer_cast<nnfusion::op::Reshape>(gnode->get_op_ptr());

    std::vector<std::vector<size_t>> out_reduce_vecs(1, std::vector<size_t>());
    if (op->get_is_transpose())
    {
        auto input_order = op->get_input_order();
        for (int d = 0; d < input_order.size(); ++d)
            out_reduce_vecs[0].push_back(in_reduce_vecs[0][input_order[d]]);
    }
    else
    {
        auto out_shape = gnode->get_output_shape(0);
        auto in_shape = gnode->get_input_shape(0);

        if (out_shape.empty())
            return std::vector<std::vector<size_t>>();

        // 1. Query the pairwise IO dims
        std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> io_pairs;
        int in_acc = 1;
        int in_index = 0;
        int out_acc = out_shape[0];
        int out_index = 0;
        std::vector<size_t> in_acc_indices;
        std::vector<size_t> out_acc_indices;
        out_acc_indices.push_back(out_index++);
        while(out_index < out_shape.size() || out_acc != 1)
        {
            if (out_acc > in_acc)
            {
                in_acc *= in_shape[in_index];
                in_acc_indices.push_back(in_index++);
            }
            else if (out_acc < in_acc)
            {
                out_acc *= out_shape[out_index];
                out_acc_indices.push_back(out_index++);
            }
            else
            {
                in_acc = 1;
                out_acc = (out_index < out_shape.size()) ? out_shape[out_index] : 1;
                io_pairs.push_back(std::make_pair(in_acc_indices, out_acc_indices));
                in_acc_indices.clear();
                out_acc_indices.clear();
                out_acc_indices.push_back(out_index++);
            }
        }

        // 2. Generate the pairwise context
        for (auto io_pair: io_pairs)
        {
            auto in_indices = io_pair.first;
            size_t shared_memory = 1;
            for (auto in_index: in_indices)
                shared_memory *= in_reduce_vecs[0][in_index];
            auto out_indices = io_pair.second;
            for (auto out_index: out_indices)
                out_reduce_vecs[0].push_back(std::min(shared_memory, out_shape[out_index]));
        }
        auto out_reduce_size = out_reduce_vecs[0].size();
        for (int e_dim = out_reduce_size; e_dim < out_shape.size(); ++e_dim)
        {
            out_reduce_vecs[0].push_back(1);
        }
    }
    return out_reduce_vecs;
}
