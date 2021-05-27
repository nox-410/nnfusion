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

//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "grid_sample.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateGridSampleOp(const onnx::NodeProto& node_proto,
                                                      const NodeMap& all_ng_nodes,
                                                      std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indices = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indices.size() == 2);
                    auto input_gnode = input_indices[0];
                    auto grid_gnode = input_indices[1];

                    Node node(node_proto);
                    int64_t mode_int = node.get_attribute_value<int64_t>("mode");
                    std::string mode;
                    switch (mode_int)
                    {
                    case 0:
                        mode = "bilinear";
                        break;
                    default:
                        NNFUSION_CHECK(false) << "Unrecognized interplation mode";
                        break;
                    }
                    int32_t align_corners = node.get_attribute_value<int64_t>("align_corners");

                    nnfusion::op::OpConfig::any op_config;
                    op_config["mode"] = mode;
                    op_config["align_corners"] = align_corners == 0 ? false : true;

                    auto grid_sample_op = std::make_shared<op::GenericOp>(node_proto.output(0), "GridSample", op_config);
                    auto grid_sample_gnode = m_graph->add_node_and_edge(grid_sample_op, {input_gnode, grid_gnode});

                    return {{node_proto.output(0), GNodeIndex(grid_sample_gnode)}};
                }
            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
