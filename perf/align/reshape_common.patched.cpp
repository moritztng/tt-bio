// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_common.hpp"

#include <string>
#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>
#include "tt_stl/small_vector.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::Shape infer_dims_for_reshape(const tt::tt_metal::Tensor& tensor, ttsl::Span<const int32_t> shape) {
    int64_t old_volume = tensor.logical_volume();
    int64_t new_volume = 1;
    int64_t index_of_negative_1 = -1;
    bool has_zero = false;
    for (auto index = 0; index < shape.size(); ++index) {
        if (shape[index] == -1) {
            if (index_of_negative_1 != -1) {
                std::string error_msg = "Shape cannot have more than 1 elements that is set to -1! Shape used: (";
                for (const auto& s : shape) {
                    error_msg += std::to_string(s) + ",";
                }
                error_msg += ")";
                TT_THROW("{}", error_msg);
            }
            index_of_negative_1 = index;
        } else {
            if (shape[index] == 0) {
                has_zero = true;
            }
            new_volume *= shape[index];
        }
    }
    if (has_zero && index_of_negative_1 != -1) {
        std::string error_msg = "cannot reshape tensor of 0 elements into shape (";
        for (const auto& s : shape) {
            error_msg += std::to_string(s) + ",";
        }
        error_msg += ") because the unspecified dimension size -1 can be any value and is ambiguous";
        TT_THROW("{}", error_msg);
    }

    ttsl::SmallVector<uint32_t> new_shape(shape.size());
    std::copy(shape.begin(), shape.end(), new_shape.begin());
    if (index_of_negative_1 == -1) {
        // Equal logical volume is the wrong invariant for a tiled tensor. A logical shape that
        // grows into padding the tensor already owns has exactly the same physical footprint --
        // same pages, same tiles, same buffer -- so it is a metadata update, not a data movement.
        // Accept that one case; everything else still requires equal volumes.
        bool widen_into_own_padding = false;
        if (new_volume > old_volume && tensor.layout() == tt::tt_metal::Layout::TILE) {
            const auto& padded = tensor.padded_shape();
            if (padded.rank() == new_shape.size() && new_shape.size() >= 2) {
                const auto tile = tensor.tensor_spec().tile();
                const uint32_t th = tile.get_height(), tw = tile.get_width();
                auto round_up = [](uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; };
                widen_into_own_padding = round_up(new_shape[new_shape.size() - 1], tw) == padded[-1] &&
                                         round_up(new_shape[new_shape.size() - 2], th) == padded[-2];
                for (size_t i = 0; i + 2 < new_shape.size(); ++i) {
                    widen_into_own_padding = widen_into_own_padding && new_shape[i] == padded[i];
                }
            }
        }
        TT_FATAL(
            new_volume == old_volume || widen_into_own_padding,
            "Invalid arguments to reshape: logical volume {} != {} and the requested shape does not "
            "fit the padding this tensor already owns (padded shape {})",
            new_volume,
            old_volume,
            tensor.padded_shape());
    } else {
        TT_FATAL(old_volume % new_volume == 0, "Invalid arguments to reshape");
        new_shape[index_of_negative_1] = old_volume / new_volume;
    }

    return tt::tt_metal::Shape(std::move(new_shape));
}

}  // namespace ttnn::operations::data_movement::detail
