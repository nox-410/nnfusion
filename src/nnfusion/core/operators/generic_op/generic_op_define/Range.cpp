// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/common/type/element_type.hpp"

REGISTER_OP(Range)
    .attr<int>("start")
    .attr<int>("limit")
    .attr<int>("delta")
    .attr<std::string>("dtype")
    .infershape(
    [](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        float start = generic_op->localOpConfig.getRoot()["start"];
        float limit = generic_op->localOpConfig.getRoot()["limit"];
        float delta = generic_op->localOpConfig.getRoot()["delta"];
        int num = (int)((limit - start + delta - 1) / delta);

        std::string dtype_str = generic_op->localOpConfig.getRoot()["dtype"];
        auto dtype = element::Type::dtype_string_to_nnfusion_element_type(dtype_str);

        nnfusion::Shape output_shape_0;
        output_shape_0.push_back(num);
        gnode->set_output_type_and_shape(0, dtype, output_shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto expr = "@output0@[N] = (@start@ + N * @delta@).cast(`@dtype@`) where N in @size@; ";

        auto generic_op = static_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        float start = generic_op->localOpConfig.getRoot()["start"];
        float delta = generic_op->localOpConfig.getRoot()["delta"];
        std::string out_dtype = generic_op->localOpConfig.getRoot()["dtype"];
        
        auto output_shape = curr->get_output_shape(0);
        // std::string out_dtype;
        // NNFUSION_CHECK(nnfusion::element::Type::nnfusion_element_type_to_dtype_string(curr->get_input_element_type(0), out_dtype));

        return op::create_code_from_template(expr, {{"start", to_string(start)}, 
                                                    {"delta", to_string(delta)},
                                                    {"size", to_string(output_shape[0])},
                                                    {"dtype", out_dtype}});
    })
    .inferrtsharedmemory([](std::shared_ptr<graph::GNode> gnode,
                            std::vector<std::vector<size_t>> in_reduce_vecs) -> std::vector<std::vector<size_t>> {
        return std::vector<std::vector<size_t>>(1, std::vector<size_t>{1});
    });
