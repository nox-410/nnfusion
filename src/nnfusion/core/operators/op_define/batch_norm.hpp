//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        class BatchNormTraining : public Op
        {
        public:
            // In this version of BatchNorm:
            //
            // MEAN AND VARIANCE: computed directly from the content of 'input'.
            //
            // OUTPUT VALUE: A tuple with the following structure:
            //   [0] - The normalization of 'input'.
            //   [1] - The per-channel means of (pre-normalized) 'input'.
            //   [2] - The per-channel variances of (pre-normalized) 'input'.
            //
            // AUTODIFF SUPPORT: yes: 'generate_adjoints(...)' works as expected.
            //
            // SHAPE DETAILS:
            //   gamma:     must have rank 1, with the same span as input's channel axis.
            //   beta:      must have rank 1, with the same span as input's channel axis.
            //   input:     must have rank >= 2.  The second dimension represents the channel axis
            //              and must have a span of at least 1.
            //   output[0]: shall have the same shape as 'input'.
            //   output[1]: shall have rank 1, with the same span as input's channel axis.
            //   output[2]: shall have rank 1, with the same span as input's channel axis.
            BatchNormTraining(double eps);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            double get_eps_value() const { return m_epsilon; }
        private:
            static constexpr size_t INPUT_GAMMA = 0;
            static constexpr size_t INPUT_BETA = 1;
            static constexpr size_t INPUT_DATA = 2;

            double m_epsilon;
        };

        class BatchNormInference : public Op
        {
        public:
            // In this version of BatchNorm:
            //
            // MEAN AND VARIANCE: provided by the 'mean' and 'variance' parameters.
            //
            // OUTPUT VALUE: a single tensor with the normalized value of 'input'.
            //
            // AUTODIFF SUPPORT:
            //   - 'generate_adjoints(...) may throw an exception.
            //
            // SHAPE DETAILS:
            //   gamma:    must have rank 1, with the same span as input's channel axis.
            //   beta:     must have rank 1, with the same span as input's channel axis.
            //   input:    must have rank >= 2. The second dimension represents the channel axis and
            //             must have a span of at least 1.
            //   mean:     must have rank 1, with the same span as input's channel axis.
            //   variance: must have rank 1, with the same span as input's channel axis.
            //   output:   shall have the same shape as 'input'.
            BatchNormInference(double eps, bool channels_first);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override;
                  
            std::vector<std::vector<size_t>> infer_runtime_share_memory(
                std::shared_ptr<graph::GNode> gnode, std::vector<std::vector<size_t>> in_reduce_vecs) override;

            bool channels_first() const { return m_channels_first; }

            double get_eps_value() const { return m_epsilon; }
        private:
            static constexpr size_t INPUT_GAMMA = 0;
            static constexpr size_t INPUT_BETA = 1;
            static constexpr size_t INPUT_DATA = 2;
            static constexpr size_t INPUT_MEAN = 3;
            static constexpr size_t INPUT_VARIANCE = 4;

            double m_epsilon;

            bool m_channels_first;
        };

        class BatchNormTrainingBackprop : public Op
        {
        public:
            BatchNormTrainingBackprop(double eps);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            double get_eps_value() const { return m_epsilon; }
        private:
            static constexpr size_t INPUT_GAMMA = 0;
            static constexpr size_t INPUT_BETA = 1;
            static constexpr size_t INPUT_DATA = 2;
            static constexpr size_t INPUT_MEAN = 3;
            static constexpr size_t INPUT_VARIANCE = 4;
            static constexpr size_t INPUT_DELTA = 5;

            double m_epsilon;
        };
    }
}
