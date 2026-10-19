// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// bmm_res writer (tt_bio/bmm_res.py): drains c_16 W tiles at a time (W divides NB). Work unit u is channel u / NS, output columns
// [j*NB, (j+1)*NB) with j = u % NS; row tile r of it is pages (c*MT + r)*NT + j*NB + [0, NB) of out. Unsplit
// (NB = NT = W) that is one contiguous row of NT pages per wait, pair_swiglu_ws's writer.
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
    constexpr uint32_t RT = get_compile_time_arg_val(4);  // rows a drain spans (BRES_RT), W / RT tiles each
    constexpr uint32_t WC = W / RT;
    constexpr uint32_t NS = NT / NB;
    constexpr uint32_t cb_out = 16;
    constexpr auto o_args = TensorAccessorArgs<5>();
    const auto so = TensorAccessor(o_args, out_addr);
    const uint32_t to = get_tile_size(cb_out);
    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t row0 = (u / NS) * MT, n0 = (u % NS) * NB;
        for (uint32_t r = 0; r < MT; r += RT) {
            for (uint32_t n = 0; n < NB; n += WC) {
                cb_wait_front(cb_out, W);
                uint32_t rd = get_read_ptr(cb_out);
#ifndef BRES_NO_WRITE  // time attribution only (af2_trimul_bmm_sweep --abl)
                for (uint32_t i = 0; i < W; ++i, rd += to) {
                    noc_async_write_page((row0 + r + i / WC) * NT + n0 + n + i % WC, so, rd);
                }
#endif
                noc_async_write_barrier();
                cb_pop_front(cb_out, W);
            }
        }
    }
}
