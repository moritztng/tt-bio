// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fp32_ln reader (tt_bio/fp32_ln.py): gamma and beta (KT full fp32 tiles each) once, then per row tile r of this
// core's nr from r0: its KT tiles of x into c_0, and twice (the mean, then rstd) compute's column-0 tile copied across
// all 32 columns into the tile compute reads back (c_4 -> c_5, c_6 -> c_7).
#include "api/dataflow/dataflow_api.h"

// dst[row][col] = src[row][0] for a 32 x 32 fp32 tile in face order (four 16 x 16 faces, row-major in each).
void broadcast_col0(uint32_t src_cb, uint32_t dst_cb) {
    cb_wait_front(src_cb, 1);
    cb_reserve_back(dst_cb, 1);
    auto* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(src_cb));
    auto* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(dst_cb));
    for (uint32_t i = 0; i < 32; ++i) {
        const uint32_t left = (i < 16 ? 0 : 512) + (i & 15) * 16;
        const uint32_t v = s[left];
        for (uint32_t j = 0; j < 16; ++j) {
            d[left + j] = v;
            d[left + 256 + j] = v;
        }
    }
    cb_push_back(dst_cb, 1);
    cb_pop_front(src_cb, 1);
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto gm = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1));
    const auto bt = TensorAccessor(b_args, get_common_arg_val<uint32_t>(2));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_x = tt::CBIndex::c_0, cb_g = tt::CBIndex::c_1, cb_b = tt::CBIndex::c_2;
    const uint32_t tb = get_tile_size(cb_x);

    cb_reserve_back(cb_g, KT);
    cb_reserve_back(cb_b, KT);
    uint32_t wg = get_write_ptr(cb_g), wb = get_write_ptr(cb_b);
    for (uint32_t t = 0; t < KT; ++t, wg += tb, wb += tb) {
        noc_async_read_page(t, gm, wg);
        noc_async_read_page(t, bt, wb);
    }
    noc_async_read_barrier();
    cb_push_back(cb_g, KT);
    cb_push_back(cb_b, KT);

    for (uint32_t r = r0; r < r0 + nr; ++r) {
        cb_reserve_back(cb_x, KT);
        uint32_t w = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < KT; ++t, w += tb) noc_async_read_page(r * KT + t, x, w);
        noc_async_read_barrier();
        cb_push_back(cb_x, KT);
        broadcast_col0(tt::CBIndex::c_4, tt::CBIndex::c_5);
        broadcast_col0(tt::CBIndex::c_6, tt::CBIndex::c_7);
    }
}
