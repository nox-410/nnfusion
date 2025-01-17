// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl.hpp"
#include "degree_based_visitor.hpp"
#include "nnfusion/engine/pass/codegen/hlsl_codegen_pass.hpp"
#include "nnfusion/engine/pass/codegen/hlsl_cpp_codegen_pass.hpp"
#include "nnfusion/engine/pass/codegen/hlsl_cs_codegen_pass.hpp"
#include "nnfusion/engine/pass/codegen/hlsl_json_codegen_pass.hpp"
#include "nnfusion/engine/pass/graph/assign_async_info_pass.hpp"
#include "nnfusion/engine/pass/graph/assign_layout_pass.hpp"
#include "nnfusion/engine/pass/graph/autodiff_pass.hpp"
#include "nnfusion/engine/pass/graph/batchnorm_inference_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/blockfusion_pass.hpp"
#include "nnfusion/engine/pass/graph/common_subexpression_elimination_pass.hpp"
#include "reversed_dfs_visitor.hpp"

#include "nnfusion/engine/pass/graph/gemm_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/gnode_device_dispatcher.hpp"
#include "nnfusion/engine/pass/graph/gradient_weight_mapping_pass.hpp"
#include "nnfusion/engine/pass/graph/ir_based_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_profiling_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/pass/graph/kernel_tuning.hpp"
#include "nnfusion/engine/pass/graph/multi_reshape_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/op_inplace_pass.hpp"
#include "nnfusion/engine/pass/graph/pattern_substitution.hpp"
#include "nnfusion/engine/pass/graph/reduce_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/manual_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/purify_graph_pass.hpp"
#include "nnfusion/engine/pass/graph/runtime_const_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/vector_dot_transpose_pass.hpp"
#include "nnfusion/engine/pass/graph/hlsl_dtype_check_pass.hpp"

#include "nnfusion/engine/pass/extract_graph_signature.hpp"
#include "nnfusion/engine/pass/tensor/inplace_tensor_analysis.hpp"
#include "nnfusion/engine/pass/tensor/liveness_analysis.hpp"
#include "nnfusion/engine/pass/tensor/tensor_device_dispatcher.hpp"
#include "nnfusion/engine/pass/tensor/tensor_memory_layout.hpp"

using namespace nnfusion;
using namespace nnfusion::engine;
using namespace nnfusion::pass::graph;
using namespace nnfusion::pass;

DEFINE_string(fhlsl_codegen_type,
              "default",
              "choose hlsl codegen type from [default(will be deprecated), csharp, cpp]");
HLSLEngine::HLSLEngine()
    : Engine()
{
    if (FLAGS_fhlsl_codegen_type == "csharp")
    {
        g_passes->push_back(make_shared<CSEPass>());
        g_passes->push_back(make_shared<AutodiffPass>());
        g_passes->push_back(make_shared<GradientWeightMappingPass>());
        g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
        g_passes->push_back(make_shared<MultiReshapeFoldingPass>());
        g_passes->push_back(make_shared<VectorDotTransposePass>());
        g_passes->push_back(make_shared<GemmFusionPass>());
        g_passes->push_back(make_shared<BatchNormInferenceFoldingPass>());
        g_passes->push_back(make_shared<AssignLayoutPass>());
        g_passes->push_back(make_shared<OpInplacePass>());
        g_passes->push_back(make_shared<ReduceFusionPass>());
        g_passes->push_back(make_shared<IRBasedFusionPass>());

        // Kernel selection
        g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
        g_passes->push_back(make_shared<KernelFusionPass>());
        g_passes->push_back(make_shared<KernelTuning>());
        g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
        g_passes->push_back(make_shared<FetchBasedSelector>());
        g_passes->push_back(make_shared<DefaultKernelSelector>());

        // Assign stream passes
        g_passes->push_back(make_shared<AssignAsyncInfoPass>());

        // Visitor
        // g_visitor = make_shared<DegreeBasedVisitor>();
        g_visitor = make_shared<ReversedDFSVisitor>();

        // extract graph signature
        m_passes->push_back(make_shared<ExtractGraphSignature>());
        // Do tensor allocation plan
        m_passes->push_back(make_shared<TensorDeviceDispatcher>());
        m_passes->push_back(make_shared<TensorLivenessAnalysis>());
        m_passes->push_back(make_shared<InplaceTensorAnalysis>());
        m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false));

        // Do codegen
        m_passes->push_back(make_shared<HLSLCSCodegenPass>());
    }
    else if (FLAGS_fhlsl_codegen_type == "cpp")
    {
        g_passes->push_back(make_shared<CSEPass>());
        g_passes->push_back(make_shared<AutodiffPass>());
        g_passes->push_back(make_shared<GradientWeightMappingPass>());
        g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
        g_passes->push_back(make_shared<MultiReshapeFoldingPass>());
        g_passes->push_back(make_shared<VectorDotTransposePass>());
        g_passes->push_back(make_shared<GemmFusionPass>());
        g_passes->push_back(make_shared<BatchNormInferenceFoldingPass>());
        g_passes->push_back(make_shared<AssignLayoutPass>());
        g_passes->push_back(make_shared<OpInplacePass>());
        g_passes->push_back(make_shared<ReduceFusionPass>());
        g_passes->push_back(make_shared<IRBasedFusionPass>());

        // Kernel selection
        g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
        g_passes->push_back(make_shared<KernelFusionPass>());
        g_passes->push_back(make_shared<KernelTuning>());
        g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
        g_passes->push_back(make_shared<FetchBasedSelector>());
        g_passes->push_back(make_shared<DefaultKernelSelector>());

        // Assign stream passes
        g_passes->push_back(make_shared<AssignAsyncInfoPass>());

        // Visitor
        // g_visitor = make_shared<DegreeBasedVisitor>();
        g_visitor = make_shared<ReversedDFSVisitor>();

        // extract graph signature
        m_passes->push_back(make_shared<ExtractGraphSignature>());
        // Do tensor allocation plan
        m_passes->push_back(make_shared<TensorDeviceDispatcher>());
        m_passes->push_back(make_shared<TensorLivenessAnalysis>());
        m_passes->push_back(make_shared<InplaceTensorAnalysis>());
        m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false));

        // Do codegen
        m_passes->push_back(make_shared<HLSLCPPCodegenPass>());
    }
    else if (FLAGS_fhlsl_codegen_type == "json")
    {
        g_passes->push_back(make_shared<GradientWeightMappingPass>());
        g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
        g_passes->push_back(make_shared<PurifyGraphPass>());
        g_passes->push_back(make_shared<HLSLDtypeCheckPass>());
        g_passes->push_back(make_shared<ManualFusionPass>());
        g_passes->push_back(make_shared<ReduceFusionPass>());
        g_passes->push_back(make_shared<PurifyGraphPass>());

        // Kernel selection
        g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
        g_passes->push_back(make_shared<KernelTuning>());
        g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
        g_passes->push_back(make_shared<FetchBasedSelector>());
        g_passes->push_back(make_shared<DefaultKernelSelector>());

        // Visitor
        g_visitor = make_shared<DegreeBasedVisitor>();

        // Do codegen
        m_passes->push_back(make_shared<HLSLJsonCodegenPass>());
    }
    else
    {
        g_passes->push_back(make_shared<GradientWeightMappingPass>());
        g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
        g_passes->push_back(make_shared<HLSLDtypeCheckPass>());
        g_passes->push_back(make_shared<ReduceFusionPass>());
        g_passes->push_back(make_shared<IRBasedFusionPass>());

        // Kernel selection
        g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
        g_passes->push_back(make_shared<KernelFusionPass>());
        g_passes->push_back(make_shared<KernelTuning>());
        g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
        g_passes->push_back(make_shared<FetchBasedSelector>());
        g_passes->push_back(make_shared<DefaultKernelSelector>());

        // Visitor
        g_visitor = make_shared<DegreeBasedVisitor>();

        // Do codegen
        m_passes->push_back(make_shared<HLSLCodegenPass>());
    }
}
