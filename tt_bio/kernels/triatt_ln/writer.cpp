// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triatt_ln writer (tt_bio/triatt_ln.py): row tile r's NW output tiles, n = c * H + h for the four
// head-major chunks q, k, v, g ([B, H, S, 32]: page ((r / MT) * H + h) * MT + r % MT, the
// HEAD_MAJOR_MT transform of kernels/triatt) and the last tile the [B, S, 32] pair bias (page r).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);    // head tiles per chunk
    constexpr uint32_t MT = get_compile_time_arg_val(1);   // row tiles per batch
    constexpr auto q_args = TensorAccessorArgs<2>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto oq = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0));
    const auto ok = TensorAccessor(k_args, get_common_arg_val<uint32_t>(1));
    const auto ov = TensorAccessor(v_args, get_common_arg_val<uint32_t>(2));
    const auto og = TensorAccessor(g_args, get_common_arg_val<uint32_t>(3));
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(4));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t NW = 4 * H + 1;
    const uint32_t tb = get_tile_size(out_cb);

    for (uint32_t r = r0; r < r0 + nrows; ++r) {
        cb_wait_front(out_cb, NW);
        uint32_t src = get_read_ptr(out_cb);
        const uint32_t base = (r / MT) * H * MT + r % MT;
        for (uint32_t h = 0; h < H; ++h, src += tb) noc_async_write_page(base + h * MT, oq, src);
        for (uint32_t h = 0; h < H; ++h, src += tb) noc_async_write_page(base + h * MT, ok, src);
        for (uint32_t h = 0; h < H; ++h, src += tb) noc_async_write_page(base + h * MT, ov, src);
        for (uint32_t h = 0; h < H; ++h, src += tb) noc_async_write_page(base + h * MT, og, src);
        noc_async_write_page(r, ob, src);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, NW);
    }
    noc_async_write_barrier();
}
