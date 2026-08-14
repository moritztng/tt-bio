// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 1's adjoint, reader side, feeding the sliding window in compute_zspread_acc.cpp.
//
// The masks are the forward's own, unchanged and resident: the adjoint of a weighted sum is the same
// weights applied to the scattered value, and there are only nplane of them.
//
// After a prologue that fills the window, each z step pushes `nstep` NEW W tiles at the back while
// the compute kernel pops `nstep` expired ones from the front. So a W tile crosses the NoC once and
// serves nplane volume tiles, which is the whole reason the adjoint's stage 1 is cheap where the
// forward's is 35.8% of the primitive. Reads are bulk 2048 B pages -- there is no per-row offset on
// this side in either direction.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(1);
    constexpr uint32_t nplane = get_compile_time_arg_val(2);
    constexpr uint32_t nstep = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t barrier_every = get_compile_time_arg_val(5);
    constexpr auto w_args = TensorAccessorArgs<6>();
    constexpr auto m_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    constexpr uint32_t nwin = nplane * nstep;

    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t nblocks = get_arg_val<uint32_t>(2);
    const uint32_t page0 = get_arg_val<uint32_t>(3);
    const uint32_t npages = get_arg_val<uint32_t>(4);

    const auto w = TensorAccessor(w_args, w_addr, tile_bytes);
    const auto m = TensorAccessor(m_args, m_addr, tile_bytes);

    cb_reserve_back(cb_mask, nplane);
    uint32_t wm = get_write_ptr(cb_mask);
    for (uint32_t p = 0; p < nplane; ++p) {
        noc_async_read_page(p, m, wm);
        wm += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_mask, nplane);

    uint32_t pg = page0;
    // Prologue: fill the window. These nwin tiles are the only ones read more than nstep at a time.
    cb_reserve_back(cb_w, nwin);
    uint32_t wp = get_write_ptr(cb_w);
    for (uint32_t i = 0; i < nwin; ++i) {
        noc_async_read_page(pg, w, wp);
        wp += tile_bytes;
        if (++pg >= npages) {
            pg = 0;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, nwin);

    // Steady state: nstep in per z step, grouped so one barrier covers barrier_every steps.
    uint32_t b = 0;
    while (b < nblocks) {
        uint32_t g = nblocks - b;
        if (g > barrier_every) {
            g = barrier_every;
        }
        cb_reserve_back(cb_w, g * nstep);
        uint32_t p = get_write_ptr(cb_w);
        for (uint32_t i = 0; i < g * nstep; ++i) {
            noc_async_read_page(pg, w, p);
            p += tile_bytes;
            if (++pg >= npages) {
                pg = 0;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, g * nstep);
        b += g;
    }
}
