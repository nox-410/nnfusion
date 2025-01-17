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
#include "nnfusion/common/coordinate.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Takes a slice of an input tensor, i.e., the sub-tensor that resides within a bounding box, optionally with stride.
        class Slice : public Op
        {
        public:
            /// \brief Constructs a tensor slice operation.
            ///
            /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
            /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
            /// \param strides The slicing strides; for example, strides of `{n,m}` means to take
            ///                every nth row and every mth column of the input matrix.
            Slice(const nnfusion::Coordinate& lower_bounds,
                  const nnfusion::Coordinate& upper_bounds,
                  const nnfusion::Strides& strides);

            /// \brief Constructs a tensor slice operation with unit strides; i.e., every element inside the bounding box will be copied to the output slice.
            ///
            /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
            /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
            Slice(const nnfusion::Coordinate& lower_bounds,
                  const nnfusion::Coordinate& upper_bounds);

            /// \return The inclusive lower-bound coordinates.
            const nnfusion::Coordinate& get_lower_bounds() const { return m_lower_bounds; }
            /// \return The exclusive upper-bound coordinates.
            const nnfusion::Coordinate& get_upper_bounds() const { return m_upper_bounds; }
            /// \return The slicing strides.
            const nnfusion::Strides& get_strides() const { return m_strides; }
            void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override;
            std::vector<std::vector<size_t>> infer_runtime_share_memory(std::shared_ptr<graph::GNode> gnode,
                std::vector<std::vector<size_t>> in_reduce_vecs) override;

        protected:
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            nnfusion::Coordinate m_lower_bounds;
            nnfusion::Coordinate m_upper_bounds;
            nnfusion::Strides m_strides;
        };
    }
}
