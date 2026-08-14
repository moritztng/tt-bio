// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fourier-slice projection stage 2 writer: one bulk page write per output tile.
//
// Bulk is the whole point. S1c measured a single 2048 B DRAM transaction at 685 ns and 382 GB/s
// chip-wide, 91% of the measured 420.2 GB/s roof, while splitting the same bytes into 64 B pieces
// falls to 30 GB/s and does not pipeline. The slice write is the design's floor (0.312 us per slice
// at box 256), so it has to run at that roof and nothing here may fragment it.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(2);

    const auto d = TensorAccessor(dst_args, dst_addr, tile_bytes);

    uint32_t page = page0;
    for (uint32_t b = 0; b < nblocks; ++b) {
        cb_wait_front(cb_out, tiles_per_block);
        uint32_t r = get_read_ptr(cb_out);
        for (uint32_t i = 0; i < tiles_per_block; ++i) {
            noc_async_write_page(page, d, r);
            r += tile_bytes;
            ++page;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, tiles_per_block);
    }
}
