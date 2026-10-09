// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail, weights resident: [z +] (o * sigmoid(g)) @ Wo. A unit is one output row
// tile (b, jt): the Kt head tiles (b, h, jt) of the head-major o and g ([B, H, S, 32], one tile per
// head), and with RESID the Nt tiles (b, jt, n) of z ([.., B, S, Nt*32]). Wo ([Kt*32, Nt*32],
// row-major tile order) goes to c_0 once and stays.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t RESID = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<4>();
    constexpr auto g_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1));
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2));
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(3));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t o_cb = tt::CBIndex::c_1;
    constexpr uint32_t g_cb = tt::CBIndex::c_2;
    constexpr uint32_t z_cb = tt::CBIndex::c_3;
    const uint32_t tb = get_tile_size(o_cb);

    cb_reserve_back(w_cb, Kt * Nt);
    uint32_t wp = get_write_ptr(w_cb);
    for (uint32_t t = 0; t < Kt * Nt; ++t, wp += tb) noc_async_read_page(t, w, wp);
    noc_async_read_barrier();
    cb_push_back(w_cb, Kt * Nt);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / St;
        const uint32_t jt = u % St;
        cb_reserve_back(g_cb, Kt);
        cb_reserve_back(o_cb, Kt);
        uint32_t gp = get_write_ptr(g_cb), op = get_write_ptr(o_cb);
        uint32_t page = b * Kt * St + jt;
        for (uint32_t h = 0; h < Kt; ++h, page += St, gp += tb, op += tb) {
            noc_async_read_page(page, g, gp);
            noc_async_read_page(page, o, op);
        }
        if constexpr (RESID) {
            cb_reserve_back(z_cb, Nt);
            uint32_t zp = get_write_ptr(z_cb);
            for (uint32_t n = 0; n < Nt; ++n, zp += tb) noc_async_read_page(u * Nt + n, z, zp);
        }
        noc_async_read_barrier();
        cb_push_back(g_cb, Kt);
        cb_push_back(o_cb, Kt);
        if constexpr (RESID) cb_push_back(z_cb, Nt);
    }
}
