// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// lead_sum reader. A TILE tensor [G, ...] holds G slabs of T tiles each; output tile t is the sum
// of input pages k*T + t over k. A core owns a contiguous range of output tiles and streams the G
// pages of each into c_0 one block of GRAN at a time. Addresses are common runtime args, as in
// rne_add, so the host caches the descriptor.
#include "api/dataflow/dataflow_api.h"

#include "../genq/genq_split.h"

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);      // slabs summed
    constexpr uint32_t T = get_compile_time_arg_val(2);      // tiles a slab

    constexpr auto s_args = TensorAccessorArgs<3>();
    const auto s = TensorAccessor(s_args, src_addr);

    constexpr uint32_t C = s_args.next_compile_time_args_offset();
    constexpr uint32_t COMPACT = get_compile_time_arg_val(C);
    uint32_t first_tile, num_tiles;
    if constexpr (COMPACT) {
        const auto sl = genq::slice<get_compile_time_arg_val(C + 1), get_compile_time_arg_val(C + 2),
                                    get_compile_time_arg_val(C + 3), get_compile_time_arg_val(C + 4),
                                    get_compile_time_arg_val(C + 5), get_compile_time_arg_val(C + 6),
                                    get_compile_time_arg_val(C + 7)>(
            get_absolute_logical_x(), get_absolute_logical_y());
        first_tile = sl.first;
        num_tiles = sl.num;
    } else {
        first_tile = get_arg_val<uint32_t>(0);
        num_tiles = get_arg_val<uint32_t>(1);
    }

    const uint32_t tile_bytes = get_tile_size(cb_in);
    for (uint32_t t = first_tile; t < first_tile + num_tiles; ++t) {
        for (uint32_t k = 0; k < G; k += GRAN) {
            const uint32_t n = (G - k < GRAN) ? (G - k) : GRAN;
            cb_reserve_back(cb_in, n);
            uint32_t w = get_write_ptr(cb_in);
            for (uint32_t j = 0; j < n; ++j) {
                noc_async_read_page((k + j) * T + t, s, w);
                w += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, n);
        }
    }
}
