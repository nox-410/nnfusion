// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Range).attr<int>("start").attr<int>("limit").attr<int>("delta").infershape(
    [](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        float start = generic_op->localOpConfig.getRoot()["start"];
        float limit = generic_op->localOpConfig.getRoot()["limit"];
        float delta = generic_op->localOpConfig.getRoot()["delta"];
        int num = (int)((limit - start + delta - 1) / delta);

        nnfusion::Shape output_shape_0;
        output_shape_0.push_back(num);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto expr = "@output0@[N] = (@start@ + N * @delta@).cast(`@dtype@`) where N in @size@";

        auto generic_op = static_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        float start = generic_op->localOpConfig.getRoot()["start"];
        float delta = generic_op->localOpConfig.getRoot()["delta"];
        
        auto output_shape = curr->get_output_shape(0);
        auto output_dtype = curr->get_input_element_type(0).c_type_string();
        if (output_dtype == "float")
            output_dtype = "float32";

        NNFUSION_CHECK(curr->get_input_element_type(0).is_real());
        return op::create_code_from_template(expr, {{"start", to_string(start)}, 
                                                    {"delta", to_string(delta)},
                                                    {"size", to_string(output_shape[0])},
                                                    {"dtype", output_dtype}});
    });
