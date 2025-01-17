// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include "generic_op.hpp"
namespace nnfusion
{
    namespace op
    {
        std::vector<std::string> create_conv_layout_from_dims(int ndims, bool is_nchw)
        {
            switch (ndims)
            {
            case 4:
                if (is_nchw)
                    return std::vector<std::string>{"N", "C", "H", "W"};
                else
                    return std::vector<std::string>{"N", "H", "W", "C"};
            case 5:
                if (is_nchw)
                    return std::vector<std::string>{"N", "C", "D", "H", "W"};
                else
                    return std::vector<std::string>{"N", "D", "H", "W", "C"};
            default:
                NNFUSION_CHECK(false) << "Unhandled dims for conv layers";
                break;
            }
        }

        std::vector<std::vector<size_t>> GenericOp::infer_runtime_share_memory(std::shared_ptr<graph::GNode> gnode, 
                                                                               std::vector<std::vector<size_t>> inputs)
        {
            if (localOpConfig.f_rt_infersharedmemory)
                return localOpConfig.f_rt_infersharedmemory(gnode, inputs);
            return std::vector<std::vector<size_t>>();
        }
    }
}