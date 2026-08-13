// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// FFT writer. Drains the output CB two tiles at a time as pass 2 produces them, so the output
// buffer is 16 tiles rather than a whole image and the write overlaps the compute that follows it.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t ntile = get_compile_time_arg_val(2);
    constexpr uint32_t nimg = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<4>();

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t page0 = get_arg_val<uint32_t>(1);

    const auto oa = TensorAccessor(o_args, o_addr, tile_bytes);

    uint32_t page = page0;
    for (uint32_t img = 0; img < nimg; ++img) {
        for (uint32_t i = 0; i < ntile; i += 2) {
            cb_wait_front(cb_o, 2);
            const uint32_t r = get_read_ptr(cb_o);
            noc_async_write_page(page + i, oa, r);
            noc_async_write_page(page + i + 1, oa, r + tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_o, 2);
        }
        page += ntile;
    }
}
