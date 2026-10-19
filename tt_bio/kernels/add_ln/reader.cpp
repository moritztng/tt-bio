// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_ln reader (tt_bio/add_ln.py). Once: gamma / beta (Kt tiles, row 0 carries the [K] vector) into c_8 / c_9,
// the 1/K reduce scaler into c_10 and an fp32 eps tile into c_11, as triatt_ln. Then per row tile r of this core's
// range, the Kt tiles of a into c_0 and of b into c_1 (pages r * Kt ..), four tiles (compute's add block) per barrier.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t SCALER = get_compile_time_arg_val(1);   // packed bf16 pair, 1/K
    constexpr uint32_t EPS = get_compile_time_arg_val(2);      // fp32 bits
    constexpr auto a_args = TensorAccessorArgs<3>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();

    const auto a = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1));
    const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(2));
    const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(3));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);

    constexpr uint32_t a_cb = tt::CBIndex::c_0;
    constexpr uint32_t b_cb = tt::CBIndex::c_1;
    constexpr uint32_t gamma_cb = tt::CBIndex::c_8;
    constexpr uint32_t beta_cb = tt::CBIndex::c_9;
    constexpr uint32_t scaler_cb = tt::CBIndex::c_10;
    constexpr uint32_t eps_cb = tt::CBIndex::c_11;
    const uint32_t tb = get_tile_size(a_cb);

    generate_reduce_scaler(scaler_cb, SCALER);
    {
        cb_reserve_back(eps_cb, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(eps_cb));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(eps_cb, 1);
    }
    cb_reserve_back(gamma_cb, Kt);
    cb_reserve_back(beta_cb, Kt);
    uint32_t w = get_write_ptr(gamma_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, gm, w);
    w = get_write_ptr(beta_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, bt, w);
    noc_async_read_barrier();
    cb_push_back(gamma_cb, Kt);
    cb_push_back(beta_cb, Kt);

    constexpr uint32_t BLK = 4;   // compute's add block
    for (uint32_t page = r0 * Kt, i = 0; i < nrows; ++i) {
        for (uint32_t i0 = 0; i0 < Kt; i0 += BLK) {
            const uint32_t n = Kt - i0 < BLK ? Kt - i0 : BLK;
            cb_reserve_back(a_cb, n);
            cb_reserve_back(b_cb, n);
            uint32_t da = get_write_ptr(a_cb), db = get_write_ptr(b_cb);
            for (uint32_t t = 0; t < n; ++t, ++page, da += tb, db += tb) {
                noc_async_read_page(page, a, da);
                noc_async_read_page(page, b, db);
            }
            noc_async_read_barrier();
            cb_push_back(a_cb, n);
            cb_push_back(b_cb, n);
        }
    }
}
