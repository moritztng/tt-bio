// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Backprojection writer: the per-row offset that the forward pass carries in its READER lives here.
//
// Section 5 said every piece of the forward transposes and that (c), the per-row integer shift, moves
// from the reader to the writer. This is that piece. The adjoint produces a 64-wide source window per
// row and each row must land at its own offset in the volume, so the bulk 2048 B page write of the
// forward becomes 32 scattered writes of 128 B.
//
// Everything measured about reads applies unchanged, because a NoC write is the same transaction: the
// offsets are quantised the same way (16 B from L1, 64 B from DRAM -- projprobe/fslice_align.py), a
// 128 B transfer costs 1.118x a 64 B one, and the cost below 512 B is per transaction rather than per
// byte. So the expected price is S1e's 32 x 128 B assembly against two bulk tile writes.
//
// `scatter` selects which, so the difference is measured rather than asserted.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t nrows = get_compile_time_arg_val(3);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(4);   // volume row pitch
    constexpr uint32_t win_bytes = get_compile_time_arg_val(5);   // bytes written per row
    constexpr uint32_t scatter = get_compile_time_arg_val(6);
    constexpr auto dst_args = TensorAccessorArgs<7>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(2);
    const uint32_t npages = get_arg_val<uint32_t>(3);

    uint32_t bo[32];
    for (uint32_t r = 0; r < nrows; ++r) {
        bo[r] = get_arg_val<uint32_t>(4 + r);
    }

    const auto d = TensorAccessor(dst_args, dst_addr,
                                  scatter ? row_bytes : tile_bytes);

    uint32_t page = page0;
    for (uint32_t b = 0; b < nblocks; ++b) {
        cb_wait_front(cb_out, out_tiles);
        const uint32_t rp = get_read_ptr(cb_out);
        if constexpr (scatter) {
            // One addressed write per row, each at its own offset into the volume.
            uint32_t src = rp;
            for (uint32_t r = 0; r < nrows; ++r) {
                uint32_t pg = page + r;
                if (pg >= npages) {
                    pg -= npages;
                }
                noc_async_write(src, d.get_noc_addr(pg, bo[r]), win_bytes);
                src += win_bytes;
            }
        } else {
            uint32_t src = rp;
            for (uint32_t i = 0; i < out_tiles; ++i) {
                uint32_t pg = page + i;
                if (pg >= npages) {
                    pg -= npages;
                }
                noc_async_write_page(pg, d, src);
                src += tile_bytes;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, out_tiles);
        page += out_tiles;
        if (page + nrows >= npages) {
            page = 0;
        }
    }
}
