// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "contrib/custom_op/custom_op.h"

REGISTER_OP(KernelPredictConv2D)
    .attr<std::string>("data_format", "NHWC")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool channels_first = op->localOpConfig.getRoot()["data_format"] == "NCHW";
        NNFUSION_CHECK(!channels_first) << "Kernel Prediction only support NHWC layout now!";
        
        NNFUSION_CHECK(2 == gnode->get_input_size());
        auto& input_shape = gnode->get_input_shape(0);
        auto& kernel_shape = gnode->get_input_shape(1);
        auto& output_shape(input_shape);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto expr_template = R"( @output0@[N, HO, WO, C] +=! @input1@[N, HO, WO, F] * @input0@[N, HO + F / 3 - 1, WO + F % 3 - 1, C].when([HO + F / 3 - 1 >= 0, HO + F / 3 - 1 < @height@, WO + F % 3 - 1 >= 0, WO + F % 3 - 1 < @width@], 0.0); )";

        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool channels_first = op->localOpConfig.getRoot()["data_format"] == "NCHW";
        NNFUSION_CHECK(!channels_first) << "Kernel Prediction only support NHWC layout now!";

        NNFUSION_CHECK(2 == gnode->get_input_size());
        auto& data_shape = gnode->get_input_shape(0);
        auto height = data_shape[1];
        auto width = data_shape[2];

        auto expr = op::create_code_from_template(expr_template, {{"height", to_string(height)},
                                                                  {"width", to_string(width)}});
        return expr;
    })
    .inferrtsharedmemory([](std::shared_ptr<graph::GNode> gnode, 
                            std::vector<std::vector<size_t>> in_reduce_vecs) -> std::vector<std::vector<size_t>>
    {
        auto in_reduce_vec_0 = in_reduce_vecs[0];
        std::vector<size_t> out_reduce_vec(in_reduce_vec_0);
        out_reduce_vec[1] = gnode->get_output_shape(0).at(1);
        out_reduce_vec[2] = gnode->get_output_shape(0).at(2);
        return std::vector<std::vector<size_t>>{out_reduce_vec};
    });
