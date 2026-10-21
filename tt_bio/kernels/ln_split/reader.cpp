// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ln_split reader (tt_bio/ln_split.py): the 1/K reduce scaler into c_10 and an fp32 eps tile into c_11, then this
// core's column group of gamma / beta (KC tiles from tile k0, row 0 carries the vector) into c_8 / c_9 and of its one
// row tile r of x (pages r * KT + k0 ..) into c_7.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KC = get_compile_time_arg_val(1);
    constexpr uint32_t SCALER = get_compile_time_arg_val(2);   // packed bf16 pair, 1/K
    constexpr uint32_t EPS = get_compile_time_arg_val(3);      // fp32 bits
    constexpr uint32_t HAS_W = get_compile_time_arg_val(4);
    constexpr uint32_t HAS_B = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto gm_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t k0 = get_arg_val<uint32_t>(1);

    constexpr uint32_t x_cb = tt::CBIndex::c_7, gamma_cb = tt::CBIndex::c_8, beta_cb = tt::CBIndex::c_9;
    constexpr uint32_t scaler_cb = tt::CBIndex::c_10, eps_cb = tt::CBIndex::c_11;
    const uint32_t tb = get_tile_size(x_cb);

    cb_reserve_back(x_cb, KC);
    uint32_t w = get_write_ptr(x_cb);
    for (uint32_t t = 0; t < KC; ++t, w += tb) noc_async_read_page(r * KT + k0 + t, x, w);
    generate_reduce_scaler(scaler_cb, SCALER);
    {
        cb_reserve_back(eps_cb, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(eps_cb));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(eps_cb, 1);
    }
    noc_async_read_barrier();
    cb_push_back(x_cb, KC);
    if constexpr (HAS_W) {
        const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(1));
        cb_reserve_back(gamma_cb, KC);
        w = get_write_ptr(gamma_cb);
        for (uint32_t t = 0; t < KC; ++t, w += tb) noc_async_read_page(k0 + t, gm, w);
    }
    if constexpr (HAS_B) {
        const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(2));
        cb_reserve_back(beta_cb, KC);
        w = get_write_ptr(beta_cb);
        for (uint32_t t = 0; t < KC; ++t, w += tb) noc_async_read_page(k0 + t, bt, w);
    }
    noc_async_read_barrier();
    if constexpr (HAS_W) cb_push_back(gamma_cb, KC);
    if constexpr (HAS_B) cb_push_back(beta_cb, KC);
}
