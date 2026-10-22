// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_swiglu_ws writer. Drains c_16, CT tiles per row tile, to pages [r*CT, (r+1)*CT) of the output; under WS_RES32
// also c_25 (the bfloat16 copy of the float32 output) to the same pages of x (common runtime arg 1), both in compute's
// two-tile fc3 groups.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 16;
    constexpr auto o_args = TensorAccessorArgs<1>();
    const auto so = TensorAccessor(o_args, out_addr);
    const uint32_t to = get_tile_size(cb_out);
#ifdef WS_RES32
    constexpr uint32_t cb_outb = 25;
    constexpr auto b_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto sb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1));
    const uint32_t tb = get_tile_size(cb_outb);
    // compute packs two tiles a fc3 group; writing each group as it lands starts the row's writes with its first
    // group instead of after its last
    constexpr uint32_t OG = CT < 2 ? CT : 2;
    for (uint32_t p = first * CT; p < (first + count) * CT; p += OG) {
        cb_wait_front(cb_out, OG);
        cb_wait_front(cb_outb, OG);
        uint32_t rd = get_read_ptr(cb_out), rb = get_read_ptr(cb_outb);
#ifndef WS_NO_WRITE
        for (uint32_t c = 0; c < OG; ++c, rd += to, rb += tb) {
            noc_async_write_page(p + c, so, rd);
            noc_async_write_page(p + c, sb, rb);
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_out, OG);
        cb_pop_front(cb_outb, OG);
    }
#else
    for (uint32_t r = first; r < first + count; ++r) {
        cb_wait_front(cb_out, CT);
        uint32_t rd = get_read_ptr(cb_out);
#ifndef WS_NO_WRITE  // time attribution only (ws_attr.py)
        for (uint32_t c = 0; c < CT; ++c, rd += to) {
            noc_async_write_page(r * CT + c, so, rd);
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_out, CT);
    }
#endif
}
