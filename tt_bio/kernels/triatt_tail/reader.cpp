// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail, weights resident: [z +] (o * sigmoid(g)) @ Wo. A unit is one output row
// tile (b, jt): the Kt head tiles (b, h, jt) of the head-major o and g ([B, H, S, 32], one tile per
// head), and with RESID the Nt tiles (b, jt, n) of z ([.., B, S, Nt*32]). Wo ([Kt*32, Nt*32],
// row-major tile order) goes to c_0 once and stays. G3D reads g as the projection wrote it,
// [.., B, S, Kt*32] (a biased block's gate, which has no head-major form), so its tile (b, jt, h).
// OBIAS loads the Nt bias tiles ([32, Nt*32], every row b) to c_7 once. TRANS (the ending block's tail
// into z^T, writer_t.cpp) walks a unit (bt, jt) as its 32 row tiles (bt*32 + r, jt), r = 0..31.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t RESID = get_compile_time_arg_val(3);
    constexpr uint32_t G3D = get_compile_time_arg_val(4);
    constexpr uint32_t OBIAS = get_compile_time_arg_val(5);
    constexpr uint32_t TRANS = get_compile_time_arg_val(6);
    constexpr auto o_args = TensorAccessorArgs<7>();
    constexpr auto g_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1));
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2));
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(3));
    const auto bias = TensorAccessor(b_args, get_common_arg_val<uint32_t>(4));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    const uint32_t rot = get_arg_val<uint32_t>(2);   // start the run rot units in, then wrap (STAGGER)

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t o_cb = tt::CBIndex::c_1;
    constexpr uint32_t g_cb = tt::CBIndex::c_2;
    constexpr uint32_t z_cb = tt::CBIndex::c_3;
    const uint32_t tb = get_tile_size(o_cb);
    const uint32_t tbg = get_tile_size(g_cb);   // bfp8 under lever triatt_ln_b8g

    cb_reserve_back(w_cb, Kt * Nt);
    uint32_t wp = get_write_ptr(w_cb);
    for (uint32_t t = 0; t < Kt * Nt; ++t, wp += tb) noc_async_read_page(t, w, wp);
    noc_async_read_barrier();
    cb_push_back(w_cb, Kt * Nt);
    if constexpr (OBIAS) {
        constexpr uint32_t b_cb = tt::CBIndex::c_7;
        cb_reserve_back(b_cb, Nt);
        uint32_t bp = get_write_ptr(b_cb);
        for (uint32_t t = 0; t < Nt; ++t, bp += tb) noc_async_read_page(t, bias, bp);
        noc_async_read_barrier();
        cb_push_back(b_cb, Nt);
    }

    for (uint32_t k = 0, v = u0 + rot; k < nunits; ++k, v = v + 1 == u0 + nunits ? u0 : v + 1) {
        const uint32_t b = TRANS ? (v >> 5) / St * 32 + (v & 31) : v / St;
        const uint32_t jt = TRANS ? (v >> 5) % St : v % St;
        const uint32_t u = b * St + jt;
        cb_reserve_back(g_cb, Kt);
        cb_reserve_back(o_cb, Kt);
        uint32_t gp = get_write_ptr(g_cb), op = get_write_ptr(o_cb);
        uint32_t page = b * Kt * St + jt;
        for (uint32_t h = 0; h < Kt; ++h, page += St, gp += tbg, op += tb) {
            noc_async_read_page(G3D ? u * Kt + h : page, g, gp);
            noc_async_read_page(page, o, op);
        }
        if constexpr (RESID) {
            cb_reserve_back(z_cb, Nt);
            uint32_t zp = get_write_ptr(z_cb);
#ifndef ABL_Z
            for (uint32_t n = 0; n < Nt; ++n, zp += tb) noc_async_read_page(u * Nt + n, z, zp);
#endif
        }
        noc_async_read_barrier();
        cb_push_back(g_cb, Kt);
        cb_push_back(o_cb, Kt);
        if constexpr (RESID) cb_push_back(z_cb, Nt);
    }
}
