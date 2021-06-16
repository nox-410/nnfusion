// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <string>
#include <set>

#include "manual_fusion_pass.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_bool(fmanual_fusion, false, "Enable manual-fusion based kernel fusion.");

class ScopeDesc
{
public:
    ScopeDesc() {};

    ScopeDesc(std::string scope_name) : scope_name(scope_name) {};

    std::string get_scope_name() { return scope_name; };

protected:
    std::string scope_name;
};

class OperatorScopeFusionOptimizer
{
public:
    OperatorScopeFusionOptimizer(std::shared_ptr<Graph> g) : 
        m_graph(g), m_scopes_desc(std::vector<ScopeDesc>()) 
    {
        m_scopes_desc.push_back(ScopeDesc("GenerateGrid"));
    };

    bool Optimize()
    {
        std::unordered_map<std::string, std::vector<std::shared_ptr<GNode>>> matched_nodes;
        for (auto node : m_graph->get_ordered_ops()) 
        {
            auto op_name = node->get_name();
            for (auto scope : m_scopes_desc)
            {
                if (node->get_op_type() == "Constant")
                    continue;
                auto pos = op_name.find(scope.get_scope_name());
                if (pos == std::string::npos)
                    continue;
                auto op_prefix = op_name.substr(0, pos);
                auto query = matched_nodes.find(op_prefix);
                if (query == matched_nodes.end())
                    matched_nodes.insert(std::make_pair(op_prefix, std::vector<std::shared_ptr<GNode>>{node}));
                else
                    query->second.push_back(node);
            }
        }

        for (auto m_node : matched_nodes)
        {
            auto subs_op = std::make_shared<nnfusion::op::Fused>(m_node.first, "Matched_Pattern");
            auto fused_node = std::make_shared<FusedGNode>(subs_op);
            std::unordered_set<std::shared_ptr<GNode>> subs_nodes;
            for (auto s_node : m_node.second)
            {
                subs_nodes.insert(s_node);
            }
            fused_node->build_fused_node(subs_nodes, m_graph);
            m_graph->add_node(fused_node);
        }

        return true;
    }

protected:
    std::shared_ptr<Graph> m_graph;
    std::vector<ScopeDesc> m_scopes_desc;
};

bool ManualFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool status = true;
    if (FLAGS_fmanual_fusion)
    {
        OperatorScopeFusionOptimizer op_scope_optimizer(graph);
        status &= op_scope_optimizer.Optimize();
    }
    return status;
}