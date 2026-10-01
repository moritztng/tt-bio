// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gate_bw reader. Three same-shaped TILE tensors (the cotangent d, the gated value o and the
// gate's pre-activation g), read page-for-page into c_0, c_1, c_2. rne_add's reader with a third
// stream: one index serves every accessor, a core owns one contiguous tile range, and the three
// addresses are common runtime args so the host caches the descriptor (rne_add's header).
#include "api/dataflow/dataflow_api.h"

#include "../genq/genq_split.h"

void kernel_main() {
    const uint32_t d_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t cb_d = 0, cb_o = 1, cb_g = 2;
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);

    constexpr auto d_args = TensorAccessorArgs<1>();
    constexpr auto o_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto sd = TensorAccessor(d_args, d_addr);
    const auto so = TensorAccessor(o_args, o_addr);
    const auto sg = TensorAccessor(g_args, g_addr);

    constexpr uint32_t G = g_args.next_compile_time_args_offset();
    constexpr uint32_t COMPACT = get_compile_time_arg_val(G);
    uint32_t first_tile, num_tiles;
    if constexpr (COMPACT) {
        const auto s = genq::slice<get_compile_time_arg_val(G + 1), get_compile_time_arg_val(G + 2),
                                   get_compile_time_arg_val(G + 3), get_compile_time_arg_val(G + 4),
                                   get_compile_time_arg_val(G + 5), get_compile_time_arg_val(G + 6),
                                   get_compile_time_arg_val(G + 7)>(
            get_absolute_logical_x(), get_absolute_logical_y());
        first_tile = s.first;
        num_tiles = s.num;
    } else {
        first_tile = get_arg_val<uint32_t>(0);
        num_tiles = get_arg_val<uint32_t>(1);
    }

    // d may be float32 (a promoted cotangent), so each stream advances by its own tile size.
    const uint32_t tile_d = get_tile_size(cb_d);
    const uint32_t tile_o = get_tile_size(cb_o);
    const uint32_t tile_g = get_tile_size(cb_g);

    uint32_t page = first_tile;
    for (uint32_t i = 0; i < num_tiles; i += GRAN) {
        const uint32_t n = (num_tiles - i < GRAN) ? (num_tiles - i) : GRAN;
        cb_reserve_back(cb_d, n);
        cb_reserve_back(cb_o, n);
        cb_reserve_back(cb_g, n);
        uint32_t wd = get_write_ptr(cb_d);
        uint32_t wo = get_write_ptr(cb_o);
        uint32_t wg = get_write_ptr(cb_g);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_page(page + j, sd, wd);
            noc_async_read_page(page + j, so, wo);
            noc_async_read_page(page + j, sg, wg);
            wd += tile_d;
            wo += tile_o;
            wg += tile_g;
        }
        noc_async_read_barrier();
        cb_push_back(cb_d, n);
        cb_push_back(cb_o, n);
        cb_push_back(cb_g, n);
        page += n;
    }
}
