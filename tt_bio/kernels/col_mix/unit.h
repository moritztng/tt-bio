// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_mix's unit of work is one head of one token tile row, for both MSA rows: unit u is tile row
// u / H and head u % H. A core owns a contiguous range of units, by genq's compact split or by
// runtime args, exactly as rne_add's kernels own a range of tiles.
#pragma once
#include <cstdint>

#include "../genq/genq_split.h"

namespace col_mix {
struct Range {
    uint32_t first, num;
};

template <uint32_t G>
inline Range units() {
    constexpr uint32_t COMPACT = get_compile_time_arg_val(G);
    if constexpr (COMPACT) {
        const auto s = genq::slice<get_compile_time_arg_val(G + 1), get_compile_time_arg_val(G + 2),
                                   get_compile_time_arg_val(G + 3), get_compile_time_arg_val(G + 4),
                                   get_compile_time_arg_val(G + 5), get_compile_time_arg_val(G + 6),
                                   get_compile_time_arg_val(G + 7)>(
            get_absolute_logical_x(), get_absolute_logical_y());
        return {s.first, s.num};
    }
    return {get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1)};
}
}  // namespace col_mix
