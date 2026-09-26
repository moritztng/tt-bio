// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rne_add writer. Drains c_16 to the output tensor page-for-page over this core`s tile range.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_tile = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = 16;  // c_16
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto sd = TensorAccessor(dst_args, dst_addr);

    const uint32_t tile_bytes = get_tile_size(cb_out);

    uint32_t page = first_tile;
    for (uint32_t i = 0; i < num_tiles; i += GRAN) {
        const uint32_t n = (num_tiles - i < GRAN) ? (num_tiles - i) : GRAN;
        cb_wait_front(cb_out, n);
        uint32_t r = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_write(r, sd.get_noc_addr(page + j), tile_bytes);
            r += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, n);
        page += n;
    }
}
