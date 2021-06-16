// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// #include "contrib/custom_op/custom_op.h"
// #include "nnfusion/frontend/util/evaluator.hpp"
// #include "nnfusion/core/kernels/cuda_gpu/cuda_cudnn.hpp"


// REGISTER_OP(BroadcastTo)
//     .attr<std::string>("T")
//     .attr<std::string>("Tidx")
//     .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
//         NNFUSION_CHECK(2 == gnode->get_input_size());

//         Shape shape;
//         NNFUSION_CHECK(nnfusion::frontend::GetValueFromNGraphOp<size_t>(gnode->get_in_edge(1)->get_src(), &shape));

//         gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), shape);
//     })
//     .antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {

//     });
