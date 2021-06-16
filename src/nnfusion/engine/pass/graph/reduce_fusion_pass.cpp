// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reduce_fusion_pass.hpp"
#include <queue>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_bool(freduce_fusion, false, "Enable reduce-range based kernel fusion.");
DEFINE_int32(freduce_range, 512, "Reduce range.");

const static int DEFAULT_GROUP_ID = -1;
namespace
{
    struct FuseGroup
    {
        FuseGroup(int g_id = DEFAULT_GROUP_ID)
            : id(g_id)
        {
        }

        std::vector<std::vector<size_t>> query_reduce_range(std::shared_ptr<GNode> node)
        {
            std::vector<std::vector<size_t>> input_range;
            for (int in_id = 0; in_id < node->get_input_size(); ++in_id)
            {
                auto in_edge = node->get_in_edge(in_id);
                auto in_node = in_edge->get_src();
                auto out_id = in_edge->get_src_output();
                auto cached = reduce_dict.find(in_node);
                if (cached != reduce_dict.end())
                    input_range.push_back(cached->second[out_id]);
                else
                    input_range.push_back(std::vector<size_t>(in_node->get_output_shape(out_id).size(), 1));
            };
            auto out_range = node->get_op_ptr()->infer_runtime_share_memory(node, input_range);
            NNFUSION_CHECK(!out_range.empty()) << "Reduce Fusion requires non-empty runtime range function definition.";
            return out_range;
        }

        bool is_empty()
        {
            return internal_nodes.empty();
        }

        void insert_root(std::shared_ptr<GNode> node)
        {
            root_node = node;
            internal_nodes.clear();
            reduce_range.clear();
            reduce_dict.clear();

            insert_node(root_node);
        }

        void insert_node(std::shared_ptr<GNode> node, 
                         std::vector<std::vector<size_t>> reduce_range=std::vector<std::vector<size_t>>())
        {
            internal_nodes.insert(node);
            NNFUSION_CHECK(reduce_range.size() == 0 || reduce_range.size() == node->get_output_size())
                << "Manually assign the reduce dict should have exactly same number with the node definition";
            if (0 == reduce_range.size())
            {
                auto out_range = query_reduce_range(node);
                reduce_dict.insert(std::make_pair(node, out_range));
            }
            else
                reduce_dict.insert(std::make_pair(node, reduce_range));
        }

        int id;
        std::unordered_set<std::shared_ptr<GNode>> internal_nodes;
        std::unordered_map<std::shared_ptr<GNode>, std::vector<std::vector<size_t>>> reduce_dict;
        std::shared_ptr<GNode> root_node;
        std::vector<size_t> reduce_range;
    };
}

class ReduceFusionOptimizer
{
public:
    ReduceFusionOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g){};

    bool optimize()
    {
        size_t before = m_graph->get_memory_io();
        ReduceFusion();
        size_t after = m_graph->get_memory_io();
        size_t update = 1;
        NNFUSION_LOG(INFO) << "[memory io] before vs after reduce fusion: " << before << " vs "
                           << after << "(update = " << update << ")";
        while (after < before)
        {
            before = after;
            ReduceFusion();
            after = m_graph->get_memory_io();
            update += 1;
            NNFUSION_LOG(INFO) << "[memory io] before vs after reduce fusion: " << before << " vs "
                               << after << "(update = " << update << ")";
        }

        return true;
    }

private:
    size_t total_reduce_range(const std::vector<size_t>& sm_vec)
    {
        NNFUSION_CHECK(!sm_vec.empty());
        size_t shared_memory = 1;
        for (size_t d : sm_vec)
            shared_memory *= d;
        return shared_memory;
    }

    std::vector<size_t> compute_reduce_range(const std::vector<size_t>& sm_a,
                                             const std::vector<size_t>& sm_b)
    {
        NNFUSION_CHECK(!sm_a.empty() && !sm_b.empty());

        std::vector<size_t> reduce_range;
        if (sm_a.size() == sm_b.size())
        {
            for (size_t i = 0; i < sm_a.size(); i++)
            {
                size_t d = sm_a[i] * sm_b[i];
                reduce_range.push_back(d);
            }
        }
        else if (total_reduce_range(sm_a) == 1)
        {
            reduce_range = sm_b;
        }
        else if (total_reduce_range(sm_b) == 1)
        {
            reduce_range = sm_a;
        }

        // NNFUSION_CHECK(!reduce_range.empty());

        return reduce_range;
    }

    bool is_fusable(std::shared_ptr<FuseGroup> fuse_group, std::shared_ptr<GNode> dst)
    {
        auto src = fuse_group->root_node;
        auto src_name = src->get_name();
        auto dst_name = dst->get_name();
        NNFUSION_CHECK_NOT_NULLPTR(src);
        if (src->get_in_edges().size() == 0 || dst->get_op_ptr()->is_output())
            return false;

        auto src_sm = src->get_op_ptr()->get_shared_memory();
        auto dst_sm = dst->get_op_ptr()->get_shared_memory();
        if (src_sm.empty() || dst_sm.empty())
            return false;

        auto group_reduce_range = fuse_group->reduce_range;
        NNFUSION_CHECK(!group_reduce_range.empty());
        auto trr = total_reduce_range(group_reduce_range);
        auto new_reduce_range = compute_reduce_range(group_reduce_range, dst_sm);
        if (new_reduce_range.empty())
            return false;
        auto new_trr = total_reduce_range(new_reduce_range);
        if (trr != new_trr && new_trr > FLAGS_freduce_range)
            return false;

        if (DFS(src, 0, dst))
        {
            fuse_group->reduce_range = new_reduce_range;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DFS(std::shared_ptr<GNode> cur, size_t step, std::shared_ptr<GNode> dst)
    {
        if (cur == dst && step > 1)
        {
            return false;
        }
        for (auto edge : cur->get_out_edges())
        {
            auto cur_dst = edge->get_dst();
            if (!cur_dst)
                continue;

            if (DFS(cur_dst, step + 1, dst))
                continue;
            else
                return false;
        }
        return true;
    }

    void ReduceFusion()
    {
        std::queue<std::shared_ptr<GNode>> ready;
        std::unordered_set<std::shared_ptr<GNode>> fused;

        for (auto node : m_graph->get_ordered_ops())
        {
            if (node->get_in_edges().size() == 0)
            {
                ready.push(node);
            }
        }

        while (!ready.empty())
        {
            auto node = ready.front();
            ready.pop();
            if (fused.find(node) != fused.end())
                continue;
            if (node->get_out_edges().size() > 1)
            {
                std::unordered_set<std::shared_ptr<GNode>> dst_set;
                for (auto edge : node->get_out_edges())
                {
                    auto dst = edge->get_dst();
                    if (dst)
                    {
                        ready.push(dst);
                        dst_set.insert(dst);
                    }
                }

                if (dst_set.size() > 1)
                    continue;
            }

            auto fuse_group = std::make_shared<FuseGroup>();
            fuse_group->internal_nodes.insert(node);
            fuse_group->reduce_range = node->get_op_ptr()->get_shared_memory();
            fuse_group->root_node = node;

            for (auto edge : node->get_out_edges())
            {
                auto dst = edge->get_dst();
                if (!dst)
                    continue;

                if (is_fusable(fuse_group, dst))
                {
                    fuse_group->internal_nodes.insert(dst);
                    fused.insert(node);
                    fused.insert(dst);
                }
                else
                {
                    ready.push(dst);
                }
            }
            if (fuse_group->internal_nodes.size() > 1)
            {
                auto fuse_node = Substitution(fuse_group);
                ready.push(fuse_node);
            }
        }
    }

    std::shared_ptr<GNode> Substitution(std::shared_ptr<FuseGroup> group)
    {
        // NNFUSION_LOG(INFO) << "-------------------group reduce range: "
        //                    << total_reduce_range(group->reduce_range);
        // for (auto node : group->internal_nodes)
        //     NNFUSION_LOG(INFO) << node->get_name();
        auto subs_op = std::make_shared<nnfusion::op::Fused>("Matched_Pattern", "Matched_Pattern");
        auto subs_node = std::make_shared<FusedGNode>(subs_op);
        subs_node->build_fused_node(group->internal_nodes, m_graph);
        m_graph->add_node(subs_node);
        subs_op->set_shared_memory(group->reduce_range);
        // NNFUSION_LOG(INFO) << "=======" << subs_node->get_name();
        return subs_node;
    }

    std::shared_ptr<Graph> m_graph;
};

class RTReduceFusionOptimizer
{
public:
    RTReduceFusionOptimizer(std::shared_ptr<Graph> graph) : m_graph(graph) {}

    bool iterative_optimize()
    {
        auto candidates = find_entry_points();

        auto fused_group = next_fused_group(candidates);
        if (fused_group->internal_nodes.size() <= 1)
            return false;

        auto fused_node = substitution(fused_group);
        NNFUSION_CHECK_NOT_NULLPTR(fused_node) << "Fused node should not be None";
        return true;
    }

    bool dfs(std::shared_ptr<GNode> cur, size_t step, 
             std::unordered_set<std::shared_ptr<GNode>> dsts)
    {
        if (dsts.find(cur) != dsts.end())
        {
            return step > 1 ? false : true;
        }
        for (auto edge : cur->get_in_edges())
        {
            auto cur_src = edge->get_src();
            if (!cur_src)
                continue;

            if (dfs(cur_src, step + 1, dsts))
                continue;
            else
                return false;
        }
        return true;
    }

    virtual std::shared_ptr<::FuseGroup> next_fused_group(std::queue<std::shared_ptr<GNode>>&) = 0;

    bool optimize()
    {
        NNFUSION_LOG(INFO) << "Reduce-based Fusion started:";
        int iter = 0;
        while(true)
        {
            size_t before = m_graph->get_memory_io();
            if(!iterative_optimize())
                break;
            size_t after = m_graph->get_memory_io();
            NNFUSION_LOG(INFO) << "[Memory access IO] before vs. after: " << before << " vs. " << after 
                               << " (update=" <<  (iter++) << ")";
        }
        NNFUSION_LOG(INFO) << "Reduce-based Fusion finished";
        return true;
    }

private:
    std::queue<std::shared_ptr<GNode>> find_entry_points()
    {
        std::queue<std::shared_ptr<GNode>> entry_points;
        for (auto node : m_graph->get_ordered_ops())
        {
            auto op = node->get_op_ptr();
            if (op->is_parameter() || op->is_output() || op->is_constant())
                continue;
            entry_points.push(node);
        }
        return entry_points;
    }
 
    std::shared_ptr<GNode> substitution(std::shared_ptr<FuseGroup> group)
    {
        // NNFUSION_LOG(INFO) << "-------------------group reduce range: "
        //                    << total_reduce_range(group->reduce_range);
        // for (auto node : group->internal_nodes)
        //     NNFUSION_LOG(INFO) << node->get_name();
        auto subs_op = std::make_shared<nnfusion::op::Fused>("Matched_Pattern", "Matched_Pattern");
        auto subs_node = std::make_shared<FusedGNode>(subs_op);
        subs_node->build_fused_node(group->internal_nodes, m_graph);
        m_graph->add_node(subs_node);
        subs_op->set_shared_memory(group->reduce_range);
        // NNFUSION_LOG(INFO) << "=======" << subs_node->get_name();
        return subs_node;
    }

    std::shared_ptr<Graph> m_graph;
    // std::unordered_set<std::shared_ptr<GNode>> m_blacklists;
};

class GreedyRTReduceFusionOptimizer : public RTReduceFusionOptimizer
{
public:
    GreedyRTReduceFusionOptimizer(std::shared_ptr<Graph> graph) : RTReduceFusionOptimizer(graph) {};

    virtual std::shared_ptr<::FuseGroup> next_fused_group(std::queue<std::shared_ptr<GNode>>& candidates) override
    {
        while(!candidates.empty())
        {
            auto fuse_group = std::make_shared<::FuseGroup>();
            std::queue<std::shared_ptr<GNode>> next_targets;
            next_targets.push(candidates.front());

            while(!next_targets.empty())
            {
                auto next = next_targets.front();
                next_targets.pop();
                if (fuse_group->internal_nodes.find(next) != fuse_group->internal_nodes.end())
                    continue;
                auto reduce_vectors = fuse_group->query_reduce_range(next);
                // Constrain the output size for now!
                bool fusable = reduce_vectors.size() > 1 ? false : true;
                for (auto r_v: reduce_vectors)
                {
                    auto reduce_range = std::accumulate(r_v.begin(), r_v.end(), 1, std::multiplies<size_t>());
                    fusable &= (reduce_range < FLAGS_freduce_range);
                }
                fusable &= dfs(next, 0, fuse_group->internal_nodes);
                if (fusable)
                {
                    fuse_group->insert_node(next, reduce_vectors);
                    if (next->get_out_edges().size() > 1)
                        continue;
                    for (auto out_edge: next->get_out_edges())
                    {
                        auto next_node = out_edge->get_dst();
                        if (next_node->get_op_ptr()->is_output())
                            continue;
                        next_targets.push(next_node);
                    }
                }
            }
            if (fuse_group->internal_nodes.size() > 1)
                return fuse_group;
            else
                candidates.pop();
        }
        return std::make_shared<::FuseGroup>();
    }
};

class ExtremeRTReduceFusionOptimizer : public RTReduceFusionOptimizer
{
public:
    ExtremeRTReduceFusionOptimizer(std::shared_ptr<Graph> graph) : RTReduceFusionOptimizer(graph) {};

    virtual std::shared_ptr<::FuseGroup> next_fused_group(std::queue<std::shared_ptr<GNode>>& candidates) override
    {
        while(!candidates.empty())
        {
            auto fuse_group = std::make_shared<::FuseGroup>();
            auto element_group = std::make_shared<::FuseGroup>();
            std::queue<std::shared_ptr<GNode>> next_targets;

            auto root = candidates.front();
            candidates.pop();

            if (root->get_out_edges().size() > 1)
                continue;
            for (auto out_edge : root->get_out_edges())
            {
                auto next_node = out_edge->get_dst();
                if (next_node->get_op_ptr()->is_output())
                    continue;
                next_targets.push(next_node);
            }

            auto reduce_range_vec = fuse_group->query_reduce_range(root).at(0);
            fuse_group->insert_root(root);
            auto pre_reduce_range = std::accumulate(reduce_range_vec.begin(), reduce_range_vec.end(), 
                1, std::multiplies<size_t>());
            while(!next_targets.empty())
            {
                auto next = next_targets.front();
                next_targets.pop();
                if (fuse_group->internal_nodes.find(next) != fuse_group->internal_nodes.end())
                    continue;
                auto reduce_vectors = element_group->query_reduce_range(next);
                // Constrain the output size for now!
                bool fusable = reduce_vectors.size() > 1 ? false : true;
                for (auto r_v: reduce_vectors)
                {
                    auto reduce_range = std::accumulate(r_v.begin(), r_v.end(), 1, std::multiplies<size_t>());
                    fusable &= (reduce_range == 1);
                }
                fusable &= dfs(next, 0, fuse_group->internal_nodes);
                if (fusable)
                {
                    fuse_group->insert_node(next, reduce_vectors);
                    if (next->get_out_edges().size() > 1)
                        continue;
                    for (auto out_edge: next->get_out_edges())
                    {
                        auto next_node = out_edge->get_dst();
                        if (next_node->get_op_ptr()->is_output())
                            continue;
                        next_targets.push(next_node);
                    }
                }
            }

            if (fuse_group->internal_nodes.size() > 1)
                return fuse_group;
        }
        return std::make_shared<::FuseGroup>();
    }
};

bool ReduceFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_freduce_fusion)
    {
        // ReduceFusionOptimizer optimizer(graph);
        auto status = GreedyRTReduceFusionOptimizer(graph).optimize();
        status &= ExtremeRTReduceFusionOptimizer(graph).optimize();
        return status;
    }
    return true;
}