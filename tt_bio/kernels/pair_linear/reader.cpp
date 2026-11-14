// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_linear reader (tt_bio/pair_linear.py), pair_swiglu_ws's reader with one weight matrix.
// The weight [KT, CT] tiles is read ONCE into c_1 and stays there for the whole call (PL_BIAS: the row-replicated
// bias, CT tiles, into c_2); then this core's row tiles of x stream through c_0, KT pages each. Row tile r is pages
// [r*KT, (r+1)*KT) of x, so nothing crosses cores and no core forwards anything to another.
//
// Common runtime args: x, w, bias addresses. Per-core runtime args: first row tile, row tile count.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t KT = get_compile_time_arg_val(0);   // K tiles of x
    constexpr uint32_t CT = get_compile_time_arg_val(1);   // output tiles
    constexpr uint32_t cb_x = 0, cb_w = 1;
    constexpr auto x_args = TensorAccessorArgs<2>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, x_addr);
    const auto sw = TensorAccessor(w_args, w_addr);
#ifdef PL_BIAS
    constexpr uint32_t cb_b = 2;
    constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto sb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(2));
#endif
    if (count == 0) {
        return;
    }

    const uint32_t tw = get_tile_size(cb_w), tx = get_tile_size(cb_x);
    cb_reserve_back(cb_w, KT * CT);
    uint32_t w = get_write_ptr(cb_w);
    for (uint32_t p = 0; p < KT * CT; ++p, w += tw) {
        noc_async_read_page(p, sw, w);
    }
#ifdef PL_BIAS
    const uint32_t tb = get_tile_size(cb_b);
    cb_reserve_back(cb_b, CT);
    w = get_write_ptr(cb_b);
    for (uint32_t p = 0; p < CT; ++p, w += tb) {
        noc_async_read_page(p, sb, w);
    }
#endif
    noc_async_read_barrier();
#ifdef PL_BIAS
    cb_push_back(cb_b, CT);
#endif
    cb_push_back(cb_w, KT * CT);

    for (uint32_t r = first; r < first + count; ++r) {
        cb_reserve_back(cb_x, KT);
        w = get_write_ptr(cb_x);
        for (uint32_t k = 0; k < KT; ++k, w += tx) {
            noc_async_read_page(r * KT + k, sx, w);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, KT);
    }
}
