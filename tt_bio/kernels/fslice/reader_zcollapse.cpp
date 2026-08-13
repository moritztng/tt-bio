// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 1 reader: the z-collapse fetches whole tiles, not per-row windows.
//
// For one 32x32 (X, Y) output tile the plane z = a*X + b*Y spans a band of z-planes, and every cell in
// the tile draws from the two planes bracketing its own z. So the tile needs every plane in the band,
// each as a full 2048 B tile at the SAME (X, Y) tile position -- coarse, fully addressed, bulk reads
// with nothing indexed per element. That is the opposite of stage 2's reader and much cheaper per byte:
// S1c measured a single 2048 B transaction at 684.7 ns from DRAM and 403.6 ns from L1, against 1,338.6
// ns for a 32 x 128 B per-row assembly of the same total size.
//
// The band width is what S3 measured over real HEALPix directions: mean 28.27 planes, p95 51.05, max
// 62.16 with the section-4.3 axis permutation in place.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_v = get_compile_time_arg_val(0);        // volume planes
    constexpr uint32_t cb_mask = get_compile_time_arg_val(1);     // per-plane weight tiles
    constexpr uint32_t nplane = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t barrier_every = get_compile_time_arg_val(4);
    // Planes entering the window per output tile. shift == nplane is the old behaviour:
    // every tile refetches a whole fresh band. shift < nplane slides the window, so a
    // plane is read once and serves nplane/shift output tiles. The z-collapse is 100%
    // reads (measured), so this ratio is the whole lever.
    constexpr uint32_t shift = get_compile_time_arg_val(5);
    constexpr auto v_args = TensorAccessorArgs<6>();
    constexpr auto m_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t v_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t nblocks = get_arg_val<uint32_t>(2);
    const uint32_t page0 = get_arg_val<uint32_t>(3);
    const uint32_t npages = get_arg_val<uint32_t>(4);

    const auto v = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto m = TensorAccessor(m_args, m_addr, tile_bytes);

    // The weight tiles are fixed for the direction, so they are loaded once rather than per tile.
    cb_reserve_back(cb_mask, nplane);
    uint32_t wm = get_write_ptr(cb_mask);
    for (uint32_t p = 0; p < nplane; ++p) {
        noc_async_read_page(p, m, wm);
        wm += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_mask, nplane);

    // Fill the window once, then top it up by `shift` planes per output tile. The circular buffer IS
    // the window: the compute pops `shift` from the front and these pushes add `shift` at the back, so
    // cb_wait_front(cb_v, nplane) always sees the current band in order.
    uint32_t next = page0;
    cb_reserve_back(cb_v, nplane);
    uint32_t w0 = get_write_ptr(cb_v);
    for (uint32_t p = 0; p < nplane; ++p) {
        uint32_t pg = next + p;
        if (pg >= npages) {
            pg -= npages;
        }
        noc_async_read_page(pg, v, w0);
        w0 += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_v, nplane);
    next += nplane;

    for (uint32_t b = 1; b < nblocks; ++b) {
        cb_reserve_back(cb_v, shift);
        uint32_t w = get_write_ptr(cb_v);
        for (uint32_t i = 0; i < shift; ++i) {
            uint32_t pg = next + i;
            if (pg >= npages) {
                pg -= npages;
            }
            noc_async_read_page(pg, v, w);
            w += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, shift);
        next += shift;
        if (next >= npages) {
            next -= npages;
        }
    }
}
