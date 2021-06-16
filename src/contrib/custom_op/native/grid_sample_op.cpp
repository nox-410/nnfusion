// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "contrib/custom_op/custom_op.h"
#include "queue"

REGISTER_OP(GridSample)
    .attr<std::string>("mode", "bilinear")
    .attr<std::string>("data_format", "NCHW")
    .attr<bool>("align_corners", false)
    .attr<bool>("norm", true)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        std::string data_format = op->localOpConfig.getRoot()["data_format"];
        NNFUSION_CHECK(data_format == "NCHW" || data_format == "NHWC");
        
        NNFUSION_CHECK(2 == gnode->get_input_size());
        auto& input_shape = gnode->get_input_shape(0);
        auto& grid_shape = gnode->get_input_shape(1);
        Shape output_shape(input_shape);
        auto offset = data_format == "NCHW" ? 1 : 0;
        output_shape[1 + offset] = grid_shape[1];
        output_shape[2 + offset] = grid_shape[2];
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);

        // std::queue<std::shared_ptr<graph::GNode>> children({gnode});
        // while (!children.empty())
        // {
        //     auto cur_node = children.front();
        //     std::cout << cur_node->get_op_type() << ": " << cur_node->get_name() << "-> " << cur_node->get_outputs()[0]->get_shape() << std::endl;
        //     for (auto in_edge : cur_node->get_in_edges())
        //         children.push(in_edge->get_src());
        //     children.pop();
        // }
    })
    .antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool chanels_first = op->localOpConfig.getRoot()["data_format"] == "NCHW";
        bool norm = op->localOpConfig.getRoot()["norm"];
        bool align_corners = op->localOpConfig.getRoot()["align_corners"];
        std::string mode = op->localOpConfig.getRoot()["mode"];
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c){ return std::tolower(c); });

        auto& input_shape = gnode->get_input_shape(0);
        auto height = chanels_first ? input_shape[2] : input_shape[1];
        auto width = chanels_first ? input_shape[3] : input_shape[2];
        
        std::vector<std::string> seq_cmds;
        
        if (mode == "bilinear" || mode == "linear")
        {
            // Un-normed Y-X coordinates
            if (!norm)
            {
                seq_cmds.push_back("mediate0[N, H, W] = @input1@[N, H, W, 0];");
                seq_cmds.push_back("mediate1[N, H, W] = @input1@[N, H, W, 1];");
            }
            else
            {
                if (align_corners)
                {
                    seq_cmds.push_back("mediate0[N, H, W] = (@input1@[N, H, W, 1] + 1) / 2 * (@height@ - 1);");
                    seq_cmds.push_back("mediate1[N, H, W] = (@input1@[N, H, W, 0] + 1) / 2 * (@width@ - 1);");
                }
                else 
                {
                    seq_cmds.push_back("mediate0[N, H, W] = ((@input1@[N, H, W, 1] + 1) * @height@ - 1) / 2;");
                    seq_cmds.push_back("mediate1[N, H, W] = ((@input1@[N, H, W, 0] + 1) * @width@ - 1) / 2;");
                }
            }

            // Corners coordinates
            seq_cmds.push_back("mediate2[N, H, W] = mediate0[N, H, W].cast(`int32`);");
            seq_cmds.push_back("mediate3[N, H, W] = mediate1[N, H, W].cast(`int32`);");
            seq_cmds.push_back("mediate4[N, H, W] = mediate0[N, H, W].cast(`int32`) + 1;");
            seq_cmds.push_back("mediate5[N, H, W] = mediate1[N, H, W].cast(`int32`) + 1;");

            auto nchw_template = "@target0@[N, C, H, W] = @input0@[N, C, @target1@[N, H, W], @target2@[N, H, W]].when([@target1@[N, H, W] >= 0, @target2@[N, H, W] >= 0, @target1@[N, H, W] < @height@, @target2@[N, H, W] < @width@], const(0.0).cast(`float32`)) * @target3@[N, H, W];";
            auto nhwc_template = "@target0@[N, H, W, C] = @input0@[N, @target1@[N, H, W], @target2@[N, H, W], C].when([@target1@[N, H, W] >= 0, @target2@[N, H, W] >= 0, @target1@[N, H, W] < @height@, @target2@[N, H, W] < @width@], const(0.0).cast(`float32`)) * @target3@[N, H, W];";
            auto corner_template = chanels_first ? nchw_template : nhwc_template;

            // Left-top corner
            seq_cmds.push_back("mediate6[N, H, W] = (mediate4[N, H, W].cast(`float32`) - mediate0[N, H, W]) * (mediate5[N, H, W].cast(`float32`) - mediate1[N, H, W]);");
            seq_cmds.push_back(op::create_code_from_template(corner_template, {{"target0", "mediate7"},
                                                                            {"target1", "mediate2"},
                                                                            {"target2", "mediate3"},
                                                                            {"target3", "mediate6"}}));

            // Right-top corner
            seq_cmds.push_back("mediate8[N, H, W] = (mediate4[N, H, W].cast(`float32`) - mediate0[N, H, W]) * (mediate1[N, H, W] - mediate3[N, H, W].cast(`float32`));");
            seq_cmds.push_back(op::create_code_from_template(corner_template, {{"target0", "mediate9"},
                                                                            {"target1", "mediate2"},
                                                                            {"target2", "mediate5"},
                                                                            {"target3", "mediate8"}}));

            // Left-bottom corner
            seq_cmds.push_back("mediate10[N, H, W] = (mediate0[N, H, W] - mediate2[N, H, W].cast(`float32`)) * (mediate5[N, H, W].cast(`float32`) - mediate1[N, H, W]);");
            seq_cmds.push_back(op::create_code_from_template(corner_template, {{"target0", "mediate11"},
                                                                            {"target1", "mediate4"},
                                                                            {"target2", "mediate3"},
                                                                            {"target3", "mediate10"}}));
            
            // Right-top corner
            seq_cmds.push_back("mediate12[N, H, W] = (mediate0[N, H, W] - mediate2[N, H, W].cast(`float32`)) * (mediate1[N, H, W] - mediate3[N, H, W].cast(`float32`));");
            seq_cmds.push_back(op::create_code_from_template(corner_template, {{"target0", "mediate13"},
                                                                            {"target1", "mediate4"},
                                                                            {"target2", "mediate5"},
                                                                            {"target3", "mediate12"}}));
            if (chanels_first)
            {
                seq_cmds.push_back("@output0@[N, C, H, W] = mediate7[N, C, H, W] + mediate9[N, C, H, W] + mediate11[N, C, H, W] + mediate13[N, C, H, W];");
            }
            else
            {
                seq_cmds.push_back("@output0@[N, H, W, C] = mediate7[N, H, W, C] + mediate9[N, H, W, C] + mediate11[N, H, W, C] + mediate13[N, H, W, C];");
            }
        }
        else if (mode == "nearest")
        {
            NNFUSION_CHECK(!chanels_first);
            seq_cmds.push_back("mediate2[N, H, W, F] = (@input1@[N, H, W, F] + 0.5).cast(`int32`);");
            seq_cmds.push_back("@output0@[N, H, W, C] = @input0@[N, mediate2[N, H, W, 0], mediate2[N, H, W, 1], C].when([mediate2[N, H, W, 0] >= 0, mediate2[N, H, W, 1] >= 0, mediate2[N, H, W, 0] < @height@, mediate2[N, H, W, 1] < @width@], const(0.0).cast(`float32`));");
        }

        auto sep = std::string(" ");
        std::string expr = join<std::vector<std::string>>(seq_cmds, sep);
        expr = op::create_code_from_template(expr, {{"height", to_string(height)},
                                                    {"width", to_string(width)}});
        return expr;
    })
    .inferrtsharedmemory([](std::shared_ptr<graph::GNode> gnode, 
                            std::vector<std::vector<size_t>> in_reduce_vecs) -> std::vector<std::vector<size_t>>
    {
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "None is not a valid value for " + gnode->get_op_type();

        // bool channels_first = op->localOpConfig.getRoot()["data_format"] == "NCHW";
        // std::string mode = op->localOpConfig.getRoot()["mode"];
        return std::vector<std::vector<size_t>>(1, std::vector<size_t>(gnode->get_output_shape(0)));
    });
