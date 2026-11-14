// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_swiglu_ws writer. Drains c_16, CT tiles per row tile, to pages [r*CT, (r+1)*CT) of the output; under WS_RES32
// also c_25 (the bfloat16 copy of the float32 output) to the same pages of x (common runtime arg 1), but not under
// WS_RES_NOCOPY, where the output is x. Under WS_ROWLIST only the host list's row tiles (common args 2-4), as the
// reader.
#include "api/dataflow/dataflow_api.h"
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
#ifdef WS_ROWLIST
    const auto rows = row_list_read<WS_ROWLIST, WS_LIST_BYTES>(first, get_common_arg_val<uint32_t>(2),
                                                               get_write_ptr(27));
    const uint32_t nrows = row_list_count(first, get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4));
    for (uint32_t i = 0; i < nrows; ++i) {
        const uint32_t r = rows[i];
#else
    for (uint32_t r = first; r < first + count; ++r) {
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
    }
}
