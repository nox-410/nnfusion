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
        // Abstract base class for elementwise arithmetic operations
        class ElementwiseArithmetic : public Op
        {
        protected:
            ElementwiseArithmetic(const std::string& node_type);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
            void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override;
            std::vector<std::vector<size_t>> infer_runtime_share_memory(std::shared_ptr<graph::GNode> gnode,
                                                                        std::vector<std::vector<size_t>>) override;
        };
    }
}
