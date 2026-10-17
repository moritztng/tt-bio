// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail T2 reader, weights resident. Once: Wp into c_1, gamma / beta (Kt tiles each, row 0
// carries the [K] vector) into c_8 / c_9, the 1/K reduce scaler into c_10 and a full fp32 eps tile
// into c_11. Then per block of Mb row tiles, in the order compute consumes them: x_out rows (c_0),
// g_out rows (c_5), z rows (c_7). z of block b is read before the writer writes block b: the
// writer only gets block b after compute popped this read, and other blocks are other rows.
// GATE 1: the g accessor is xb (the trimul's input norm, Kt tiles per row tile) and goes to c_6; Wg
// follows Wp in c_1 (compute_ln.cpp computes g itself).
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Mb = get_compile_time_arg_val(2);
    constexpr uint32_t SCALER = get_compile_time_arg_val(3);   // packed bf16 pair
    constexpr uint32_t EPS = get_compile_time_arg_val(4);      // fp32 bits
    constexpr uint32_t GATE = get_compile_time_arg_val(5);
    // T2_ABL bit 16 (bench only): the per-block reads are skipped and the CBs pushed as they are.
    constexpr bool NOREAD = get_compile_time_arg_val(6) & 16;
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();
    constexpr auto wg_args = TensorAccessorArgs<bt_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto wp = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1));
    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2));
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(3));
    const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(4));
    const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(5));
    const auto wg = TensorAccessor(wg_args, get_common_arg_val<uint32_t>(6));
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t g_cb = GATE ? tt::CBIndex::c_6 : tt::CBIndex::c_5;
    constexpr uint32_t gt = GATE ? Kt : Nt;                     // g / xb tiles per row tile
    constexpr uint32_t z_cb = tt::CBIndex::c_7;
    constexpr uint32_t gamma_cb = tt::CBIndex::c_8;
    constexpr uint32_t beta_cb = tt::CBIndex::c_9;
    constexpr uint32_t scaler_cb = tt::CBIndex::c_10;
    constexpr uint32_t eps_cb = tt::CBIndex::c_11;
    const uint32_t tb = get_tile_size(in0_cb);
    const uint32_t tg = get_tile_size(g_cb);                    // bf16, or bfp8 g_out (trimul_b8in)

    generate_reduce_scaler(scaler_cb, SCALER);
    {
        cb_reserve_back(eps_cb, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(eps_cb));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(eps_cb, 1);
    }
    cb_reserve_back(in1_cb, (1 + GATE) * Kt * Nt);
    cb_reserve_back(gamma_cb, Kt);
    cb_reserve_back(beta_cb, Kt);
    uint32_t w = get_write_ptr(in1_cb);
    for (uint32_t t = 0; t < Kt * Nt; ++t, w += tb) noc_async_read_page(t, wp, w);
    if constexpr (GATE) {
        for (uint32_t t = 0; t < Kt * Nt; ++t, w += tb) noc_async_read_page(t, wg, w);
    }
    w = get_write_ptr(gamma_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, gm, w);
    w = get_write_ptr(beta_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, bt, w);
    noc_async_read_barrier();
    cb_push_back(in1_cb, (1 + GATE) * Kt * Nt);
    cb_push_back(gamma_cb, Kt);
    cb_push_back(beta_cb, Kt);

    for (uint32_t b = 0; b < nblocks; ++b) {
        const uint32_t m = m0 + b * Mb;
        cb_reserve_back(in0_cb, Mb * Kt);
        w = get_write_ptr(in0_cb);
        for (uint32_t t = 0; t < (NOREAD ? 0 : Mb * Kt); ++t, w += tb) noc_async_read_page(m * Kt + t, x, w);
        noc_async_read_barrier();
        cb_push_back(in0_cb, Mb * Kt);

        cb_reserve_back(g_cb, Mb * gt);
        w = get_write_ptr(g_cb);
        for (uint32_t t = 0; t < (NOREAD ? 0 : Mb * gt); ++t, w += tg) noc_async_read_page(m * gt + t, g, w);
        cb_reserve_back(z_cb, Mb * Nt);
        uint32_t wz = get_write_ptr(z_cb);
        for (uint32_t t = 0; t < (NOREAD ? 0 : Mb * Nt); ++t, wz += tb) noc_async_read_page(m * Nt + t, z, wz);
        noc_async_read_barrier();
        cb_push_back(g_cb, Mb * gt);
        cb_push_back(z_cb, Mb * Nt);
    }
}
