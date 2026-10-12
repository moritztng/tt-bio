// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Transition T3 reader. Once: the resident weights W12 = [W1 | W2] ([Kt, 2*Ht] tiles) and W3 ([Ht, Kt]),
// gamma, beta, the 1/K reduce scaler and eps. Then per unit (one row tile of x: Kt contiguous pages,
// page u * Kt + dt) the Kt tiles into the LayerNorm input and, copied core-locally, into the residual CB.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t RES = get_compile_time_arg_val(2);      // 1: the residual CB is fed
    constexpr uint32_t SCALER = get_compile_time_arg_val(3);   // packed bf16 pair, 1/K
    constexpr uint32_t EPS = get_compile_time_arg_val(4);      // fp32 bits
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto w12_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w3_args = TensorAccessorArgs<w12_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<w3_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto w12 = TensorAccessor(w12_args, get_common_arg_val<uint32_t>(1));
    const auto w3 = TensorAccessor(w3_args, get_common_arg_val<uint32_t>(2));
    const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(3));
    const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(4));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t w12_cb = tt::CBIndex::c_0;
    constexpr uint32_t w3_cb = tt::CBIndex::c_1;
    constexpr uint32_t res_cb = tt::CBIndex::c_5;
    constexpr uint32_t x_cb = tt::CBIndex::c_7;
    constexpr uint32_t gamma_cb = tt::CBIndex::c_8;
    constexpr uint32_t beta_cb = tt::CBIndex::c_9;
    constexpr uint32_t scaler_cb = tt::CBIndex::c_10;
    constexpr uint32_t eps_cb = tt::CBIndex::c_11;
    const uint32_t tb = get_tile_size(x_cb);

    generate_reduce_scaler(scaler_cb, SCALER);
    {
        cb_reserve_back(eps_cb, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(eps_cb));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(eps_cb, 1);
    }
    cb_reserve_back(w12_cb, 2 * Ht * Kt);
    cb_reserve_back(w3_cb, Ht * Kt);
    cb_reserve_back(gamma_cb, Kt);
    cb_reserve_back(beta_cb, Kt);
    // The weights are bf16 or bfp8 (`transition_b8`): their CBs carry the tensor's format.
    const uint32_t tw = get_tile_size(w12_cb);
    uint32_t w = get_write_ptr(w12_cb);
    for (uint32_t t = 0; t < 2 * Ht * Kt; ++t, w += tw) noc_async_read_page(t, w12, w);
    w = get_write_ptr(w3_cb);
    for (uint32_t t = 0; t < Ht * Kt; ++t, w += tw) noc_async_read_page(t, w3, w);
    w = get_write_ptr(gamma_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, gm, w);
    w = get_write_ptr(beta_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, bt, w);
    noc_async_read_barrier();
    cb_push_back(w12_cb, 2 * Ht * Kt);
    cb_push_back(w3_cb, Ht * Kt);
    cb_push_back(gamma_cb, Kt);
    cb_push_back(beta_cb, Kt);

    uint32_t page = u0 * Kt;
    for (uint32_t u = 0; u < nunits; ++u) {
        cb_reserve_back(x_cb, Kt);
        const uint32_t dst = get_write_ptr(x_cb);
        for (uint32_t dt = 0; dt < Kt; ++dt, ++page) noc_async_read_page(page, x, dst + dt * tb);
        noc_async_read_barrier();
        if constexpr (RES) {
            cb_reserve_back(res_cb, Kt);
            noc_async_read(get_noc_addr(dst), get_write_ptr(res_cb), Kt * tb);
            noc_async_read_barrier();
            cb_push_back(res_cb, Kt);
        }
        cb_push_back(x_cb, Kt);
    }
}
