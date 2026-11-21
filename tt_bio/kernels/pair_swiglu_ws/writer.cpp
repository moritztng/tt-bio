// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_swiglu_ws writer. Drains c_16, CT tiles per row tile, to pages [r*CT, (r+1)*CT) of the output; under WS_RES32
// also c_25 (the bfloat16 copy of the float32 output) to the same pages of x (common runtime arg 1), but not under
// WS_RES_NOCOPY, where the output is x. Under WS_ROWLIST only the host list's row tiles (common args 2-4), as the
// reader.
#include "api/dataflow/dataflow_api.h"
#include "../common/pb_order.hpp"
#ifdef WS_ROWLIST
#include "../common/row_list.hpp"
#endif

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 16;
    constexpr auto o_args = TensorAccessorArgs<1>();
    const auto so = TensorAccessor(o_args, out_addr);
    const uint32_t to = get_tile_size(cb_out);
#if defined(WS_RES32) && !defined(WS_RES_NOCOPY)
    constexpr uint32_t cb_outb = 25;
    constexpr auto b_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto sb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1));
    const uint32_t tb = get_tile_size(cb_outb);
#endif
#ifdef WS_PB
    // WS_PB: compile args from WS_PB_ARGS: NT, HO (heads per output), HV (real heads), the bias accessor; common args
    // from 2: one address per output [B, HV, n, n]. Head h of the 32 goes to output h / HO, head h % HO, if real.
    constexpr uint32_t PB_NT = get_compile_time_arg_val(WS_PB_ARGS), PB_HO = get_compile_time_arg_val(WS_PB_ARGS + 1),
                       PB_HV = get_compile_time_arg_val(WS_PB_ARGS + 2), cb_pb = 30;
    constexpr auto pb_args = TensorAccessorArgs<WS_PB_ARGS + 3>();
#endif
#ifdef WS_ROWLIST
    const auto rows = row_list_read<WS_ROWLIST, WS_LIST_BYTES>(first, get_common_arg_val<uint32_t>(2),
                                                               get_write_ptr(27));
    const uint32_t nrows = row_list_count(first, get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4));
    for (uint32_t i = 0; i < nrows; ++i) {
        const uint32_t r = rows[i];
#else
    for (uint32_t q = first; q < first + count; ++q) {
        const uint32_t r = ws_row(q);
#endif
        cb_wait_front(cb_out, CT);
        uint32_t rd = get_read_ptr(cb_out);
#ifndef WS_NO_WRITE  // time attribution only (ws_attr.py)
        for (uint32_t c = 0; c < CT; ++c, rd += to) {
            noc_async_write_page(r * CT + c, so, rd);
        }
#endif
#if defined(WS_RES32) && !defined(WS_RES_NOCOPY)
        cb_wait_front(cb_outb, CT);
        uint32_t rb = get_read_ptr(cb_outb);
        for (uint32_t c = 0; c < CT; ++c, rb += tb) {
            noc_async_write_page(r * CT + c, sb, rb);
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_out, CT);
#if defined(WS_RES32) && !defined(WS_RES_NOCOPY)
        cb_pop_front(cb_outb, CT);
#endif
#ifdef WS_PB
        if ((q & 31) == 31) {   // the group's 32 head tiles, as kernels/pair_bias_rms/writer.cpp places them
            const uint32_t u = q >> 5, b = u / (PB_NT * PB_NT), ib = (u / PB_NT) % PB_NT, jb = u % PB_NT;
            for (uint32_t h = 0; h < 32; ++h) {
                cb_wait_front(cb_pb, 1);
                if (h % PB_HO < PB_HV) {
                    const auto sp = TensorAccessor(pb_args, get_common_arg_val<uint32_t>(2 + h / PB_HO));
                    noc_async_write_page(((b * PB_HV + h % PB_HO) * PB_NT + ib) * PB_NT + jb, sp, get_read_ptr(cb_pb));
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_pb, 1);
            }
        }
#endif
    }
#ifdef WS_PB
    noc_async_write_barrier();
#endif
}
