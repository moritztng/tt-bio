// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_ln_transpose reader: LayerNorm(x)^T for x [S1, S2, C] in one pass. A unit is the 32 i of It by the
// 32 j of Jt: its 32 row tiles (i, Jt), Kt channel tiles each, stream through c_5 in i order (pages
// (i * S2t + Jt) * Kt + k). gamma / beta (Kt tiles, row 0), the 1/K scaler and an fp32 eps tile are read
// once into c_8 / c_9 / c_10 / c_11, as pair_swiglu_ws's WS_LN reader.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t S2t = get_compile_time_arg_val(1);
    constexpr uint32_t SCALER = get_compile_time_arg_val(2);
    constexpr uint32_t EPS = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto sg = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1));
    const auto sb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(2));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    if (nunits == 0) {
        return;
    }
    constexpr uint32_t cb_x = 5, cb_gamma = 8, cb_beta = 9, cb_scaler = 10, cb_eps = 11;
    const uint32_t tx = get_tile_size(cb_x);

    generate_reduce_scaler(cb_scaler, SCALER);
    cb_reserve_back(cb_eps, 1);
    volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
    for (uint32_t i = 0; i < 1024; ++i) e[i] = EPS;
    cb_push_back(cb_eps, 1);
    cb_reserve_back(cb_gamma, Kt);
    cb_reserve_back(cb_beta, Kt);
    uint32_t w = get_write_ptr(cb_gamma);
    for (uint32_t p = 0; p < Kt; ++p, w += tx) noc_async_read_page(p, sg, w);
    w = get_write_ptr(cb_beta);
    for (uint32_t p = 0; p < Kt; ++p, w += tx) noc_async_read_page(p, sb, w);
    noc_async_read_barrier();
    cb_push_back(cb_gamma, Kt);
    cb_push_back(cb_beta, Kt);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t It = u / S2t;
        const uint32_t Jt = u % S2t;
        for (uint32_t r = 0; r < 32; ++r) {
            cb_reserve_back(cb_x, Kt);
            w = get_write_ptr(cb_x);
            const uint32_t p0 = ((It * 32 + r) * S2t + Jt) * Kt;
            for (uint32_t k = 0; k < Kt; ++k, w += tx) noc_async_read_page(p0 + k, sx, w);
            noc_async_read_barrier();
            cb_push_back(cb_x, Kt);
        }
    }
}
