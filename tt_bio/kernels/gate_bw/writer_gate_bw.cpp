// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gate_bw writer. Drains c_16 (do) and c_17 (dg) to the two gradients page-for-page over this
// core's tile range; rne_add's writer with a second stream.
#include "api/dataflow/dataflow_api.h"

#include "../genq/genq_split.h"

void kernel_main() {
    const uint32_t do_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t dg_addr = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t cb_do = 16, cb_dg = 17;
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);

    constexpr auto do_args = TensorAccessorArgs<1>();
    constexpr auto dg_args = TensorAccessorArgs<do_args.next_compile_time_args_offset()>();
    const auto s_do = TensorAccessor(do_args, do_addr);
    const auto s_dg = TensorAccessor(dg_args, dg_addr);

    constexpr uint32_t G = dg_args.next_compile_time_args_offset();
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

    const uint32_t tile_do = get_tile_size(cb_do);
    const uint32_t tile_dg = get_tile_size(cb_dg);

    uint32_t page = first_tile;
    for (uint32_t i = 0; i < num_tiles; i += GRAN) {
        const uint32_t n = (num_tiles - i < GRAN) ? (num_tiles - i) : GRAN;
        cb_wait_front(cb_do, n);
        cb_wait_front(cb_dg, n);
        uint32_t r_do = get_read_ptr(cb_do);
        uint32_t r_dg = get_read_ptr(cb_dg);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_write(r_do, s_do.get_noc_addr(page + j), tile_do);
            noc_async_write(r_dg, s_dg.get_noc_addr(page + j), tile_dg);
            r_do += tile_do;
            r_dg += tile_dg;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_do, n);
        cb_pop_front(cb_dg, n);
        page += n;
    }
}
