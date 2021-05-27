// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl_dtype_check_pass.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool HLSLDtypeCheckPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    for (auto node : graph->get_ordered_ops())
    {
        for (auto output : node->get_outputs())
        {
            if(output->get_element_type() == nnfusion::element::i8)
                output->set_type_and_shape(nnfusion::element::i32, output->get_shape());
            else if (output->get_element_type() == nnfusion::element::u8)
                output->set_type_and_shape(nnfusion::element::i32, output->get_shape());
            else if (output->get_element_type() == nnfusion::element::boolean)
                output->set_type_and_shape(nnfusion::element::i32, output->get_shape());
        }
        for (auto in_id = 0; in_id < node->get_input_size(); ++in_id)
        {
            auto in_edge = node->get_in_edge(in_id);
            if (in_edge->is_control_edge())
                continue;
            auto src_output = in_edge->get_src()->get_outputs().at(in_edge->get_src_output());
            node->get_inputs().at(in_id)->set_element_type(src_output->get_element_type());

            auto input = node->get_inputs().at(in_id);

            NNFUSION_CHECK(input->get_element_type() != nnfusion::element::i8 && input->get_element_type() != nnfusion::element::u8 && input->get_element_type() != nnfusion::element::boolean) << "Unsupport inputs dtype with uint8/int8/char for HLSL codegen";
        }
    }
    return true;
}
