// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdexcept>
#include <stdio.h>

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class Empty : public CudaEmitter
            {
            public:
                Empty(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {

                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override
                {
                    m_gridDim = dim3(1, 1, 1);
                    m_blockDim = dim3(1, 1, 1);
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<nnfusion::op::Variable> op;
                size_t threads;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("SpaceToDepth",                                                //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), //attrs
                        cuda::Empty)                                            // constructor

REGISTER_KERNEL_EMITTER("DepthToSpace",                                                //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), //attrs
                        cuda::Empty)

REGISTER_KERNEL_EMITTER("DepthwiseConv1dNative",                                                //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), //attrs
                        cuda::Empty)

REGISTER_KERNEL_EMITTER("Resize",                                                //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), //attrs
                        cuda::Empty)

