// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// bmm_res writer (tt_bio/bmm_res.py): drains c_16 W tiles at a time (W divides NB). Work unit u is channel u / NS, output columns
// [j*NB, (j+1)*NB) with j = u % NS; row tile r of it is pages (c*MT + r)*NT + j*NB + [0, NB) of out. Unsplit
// (NB = NT = W) that is one contiguous row of NT pages per wait, pair_swiglu_ws's writer.
//
// BRES_ROT (bmm_res.matmul(rot=)): channel c's page q goes to c*MT*NT + (q + c) % (MT*NT), so a channel's tile (i, j)
// moves one bank per channel and a reader of one (i, j) across channels spreads over every bank.
//
// Common runtime args: out address. Per-core runtime args: first unit, unit count.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);
    constexpr uint32_t NS = NT / NB;
    constexpr uint32_t cb_out = 16;
    constexpr auto o_args = TensorAccessorArgs<4>();
    const auto so = TensorAccessor(o_args, out_addr);
    const uint32_t to = get_tile_size(cb_out);
    constexpr uint32_t CH = MT * NT;
    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t row0 = (u / NS) * MT, n0 = (u % NS) * NB;
#ifdef BRES_ROT
        const uint32_t rot = (u / NS) % CH;
#endif
        for (uint32_t r0 = 0; r0 < MT; ++r0) {
#ifdef BRES_RSTAG
            const uint32_t r = (r0 + u / NS) % MT;
#else
            const uint32_t r = r0;
#endif
            for (uint32_t n = 0; n < NB; n += W) {
                cb_wait_front(cb_out, W);
                uint32_t rd = get_read_ptr(cb_out);
#ifndef BRES_NO_WRITE  // time attribution only (af2_trimul_bmm_sweep --abl)
                for (uint32_t i = 0; i < W; ++i, rd += to) {
#ifdef BRES_ROT
                    const uint32_t q = r * NT + n0 + n + i + rot;
                    noc_async_write_page(row0 * NT + (q < CH ? q : q - CH), so, rd);
#else
                    noc_async_write_page((row0 + r) * NT + n0 + n + i, so, rd);
#endif
                }
#endif
                noc_async_write_barrier();
                cb_pop_front(cb_out, W);
            }
        }
    }
}
