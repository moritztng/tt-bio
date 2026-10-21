// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// addcmul_at writer: drains c_16 to the (dense) output page-for-page over this core's tile range.
#include "api/dataflow/dataflow_api.h"

#include "../genq/genq_split.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t G = 1;  // genq compact split, see the reader
    uint32_t first_tile, num_tiles;
    if constexpr (get_compile_time_arg_val(G) == 1) {
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
    constexpr auto dst_args = TensorAccessorArgs<G + 8>();
    const auto sd = TensorAccessor(dst_args, dst_addr);
    constexpr uint32_t cb_out = 16;
    const uint32_t tile = get_tile_size(cb_out);

    uint32_t page = first_tile;
    for (uint32_t i = 0; i < num_tiles; i += GRAN) {
        const uint32_t n = (num_tiles - i < GRAN) ? (num_tiles - i) : GRAN;
        cb_wait_front(cb_out, n);
        uint32_t rp = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_write(rp, sd.get_noc_addr(page + j), tile);
            rp += tile;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, n);
        page += n;
    }
}
