// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/common.hpp"
#include "hlsl_json_codegen_pass.hpp"
#include "nnfusion/core/kernels/hlsl/hlsl_kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::op;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::codegen;
using namespace nnfusion::element;

DECLARE_string(fdefault_device);

void HLSLJsonCodegenPass::initialize(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    // setup lup_codegen execution info
    projgen->lup_codegen->pwd = m_codegen_folder;
    projgen->lup_codegen->write_to = "nnfusion_rt.json";
    return;
}

bool HLSLJsonCodegenPass::collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;

    auto lup_func_calls = get_kernel_func_calls("func_calls", projgen->lup_exec);

    auto& graph = tu->m_graph;
    auto& prog = tu->program;

    OpConfig::any kernel_defs({{"Parameter", std::vector<OpConfig::any>()}, 
                               {"Result", std::vector<OpConfig::any>()},
                               {"Constant", std::vector<OpConfig::any>()},
                               {"Mediate", std::vector<OpConfig::any>()},
                               {"Op", std::vector<OpConfig::any>()}});

    for (auto iterator : prog)
    {
        for (auto ins : *iterator)
        {
            auto gnode = ins->getGNode();
            auto kernel = ins->getKernel();
            auto gnode_ctx = kernel->m_context;
   
            // process kernel code
            FunctionUnit_p fu = kernel->get_or_emit_source(true);
            string call_str = fu->call_unit->get_code();
            string body_str = fu->body_unit->get_code();

            std::vector<OpConfig::any> nodes;
            for (int i = 0; i < gnode_ctx->outputs.size(); ++i)
            {
                OpConfig::any node;
                node["name"] = "ts_" + gnode_ctx->outputs[0]->get_name();
                node["shape"] = gnode->get_shape();
                std::string dtype_str; 
                Type::nnfusion_element_type_to_dtype_string(gnode->get_output_element_type(i), dtype_str);
                node["dtype"] = dtype_str;
                nodes.push_back(node);
            }
            if (gnode->get_op_type() == "Parameter")
            {
                NNFUSION_CHECK(nodes.size() == 1);
                kernel_defs["Parameter"].push_back(nodes[0]);
            }
            else if (gnode->get_op_type() == "Constant")
            {
                NNFUSION_CHECK(nodes.size() == 1);
                nodes[0]["source"] = gnode_ctx->outputs[0]->get_name() + ".bin";
                kernel_defs["Constant"].push_back(nodes[0]);
            }
            else if (gnode->get_op_type() == "Result")
            {
                OpConfig::any result;
                result["name"] = "ts_" + gnode_ctx->inputs[0]->get_name();
                kernel_defs["Result"].push_back(result);
            }
            else
            {
                OpConfig::any operate;
                operate["name"] = "op_" + gnode_ctx->outputs[0]->get_name();
                operate["raw_name"] = gnode->get_name();

                auto kernel_func_def = kernel_func_defs.find(body_str);
                if(kernel_func_def == kernel_func_defs.end())
                {
                    std::string source;
                    if (kernel->get_function_name().length() > 128)
                    {
                        size_t hashcode = std::hash<std::string>{}(kernel->get_function_name());
                        source = "compressed_src_" + std::to_string(hashcode) + ".hlsl";
                    }
                    else
                        source = kernel->get_function_name() + ".hlsl";
                    operate["source"] = source;

                    auto func_def = fu->body_unit;
                    for (auto& it : fu->dep_unit->local_symbol)
                    {
                        func_def->require(it.second);
                    }

                    kernel_func_defs[body_str] = make_pair(source, func_def);
                    lup_func_calls->require(kernel_func_defs[body_str].second);
                }
                else
                    operate["source"] = kernel_func_def->second.first;
                std::vector<std::string> inputs;
                for (int i = 0; i < gnode_ctx->input_names.size(); ++i)
                    inputs.push_back("ts_" + gnode_ctx->input_names[i]);
                operate["inputs"] = inputs;

                std::vector<std::string> outputs;
                for (int i = 0; i < gnode_ctx->output_names.size(); ++i)
                {
                    outputs.push_back("ts_" + gnode_ctx->output_names[i]);
                    kernel_defs["Mediate"].push_back(nodes[i]);
                }
                operate["outputs"] = outputs;
                kernel_defs["Op"].push_back(operate);
            }

        }
    }

    std::string paramter_json = kernel_defs.dump(4);
    LanguageUnit_p parameter_body = std::make_shared<LanguageUnit>("body", paramter_json);
    lup_func_calls->unit_vec.push_back(parameter_body);

    LanguageUnit_p end = std::make_shared<LanguageUnit>("end", "\n");
    lup_func_calls->unit_vec.push_back(end);

    separate_func_defs_files(-1, m_kernel_folder);

    return true;
}