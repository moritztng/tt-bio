// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triatt_ln reader (tt_bio/triatt_ln.py). Once: the [K, NW] qkvgb weight into c_0 (resident),
// gamma / beta (Kt tiles, row 0 carries the [K] vector) into c_8 / c_9, the 1/K reduce scaler into
// c_10 and an fp32 eps tile into c_11, as trimul T1. Then this core's row tiles r0 .. r0 + n - 1 of
// the raw pair x, Kt tiles each (row tile r is pages r * Kt ..), RB row tiles per barrier, into c_7.
// TR (runtime arg 2): x is z and the kernel norms z^T without writing it. A unit u = (bt, jt, half) is the 16
// row tiles b = bt*32 + half*16 + k (k < 16) of z^T at column tile jt; their rows are rows half*16 + k of the 32
// z tiles (s = jt*32 + q, bt), i.e. faces 2*half and 2*half+1 of each, one contiguous KB. Per channel tile the
// 32 half tiles land in a c_3 stage and are shuffled as pair_transpose's writer does: row k of stage tile q ->
// row q of row tile k. Writer and compute see the same rows in the same order (writer.cpp row_of).
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "../common/ln_rows.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t NW = get_compile_time_arg_val(1);
    constexpr uint32_t RB = get_compile_time_arg_val(2);
    constexpr uint32_t SCALER = get_compile_time_arg_val(3);   // packed bf16 pair, 1/K
    constexpr uint32_t EPS = get_compile_time_arg_val(4);      // fp32 bits
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto wt = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1));
    const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(2));
    const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(3));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
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
    cb_reserve_back(w_cb, Kt * NW);
    cb_reserve_back(gamma_cb, Kt);
    cb_reserve_back(beta_cb, Kt);
    uint32_t w = get_write_ptr(w_cb);
    for (uint32_t t = 0; t < Kt * NW; ++t, w += tb) noc_async_read_page(t, wt, w);
    w = get_write_ptr(gamma_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, gm, w);
    w = get_write_ptr(beta_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, bt, w);
    noc_async_read_barrier();
    cb_push_back(w_cb, Kt * NW);
    replicate_row0(get_write_ptr(gamma_cb), Kt, get_tile_size(gamma_cb));
    replicate_row0(get_write_ptr(beta_cb), Kt, get_tile_size(beta_cb));
    cb_push_back(gamma_cb, Kt);
    cb_push_back(beta_cb, Kt);

    if (get_arg_val<uint32_t>(2)) {
        constexpr uint32_t ROW = 32, FACE = 16 * ROW, HALF = 2 * FACE;
        const uint32_t MT = get_arg_val<uint32_t>(3);        // row tiles per b = column tiles of z
        const uint32_t stage = get_write_ptr(tt::CBIndex::c_3);
        auto row = [](uint32_t r) { return ((r >> 4) * 2) * FACE + (r & 15) * ROW; };
        for (uint32_t u = r0; u < r0 + nrows; ++u) {
            const uint32_t half = u & 1, jt = (u >> 1) % MT, bt = (u >> 1) / MT;
            cb_reserve_back(x_cb, 16 * Kt);
            const uint32_t dst = get_write_ptr(x_cb);
            for (uint32_t n = 0; n < Kt; ++n) {
                for (uint32_t q = 0; q < 32; ++q)
                    noc_async_read(x.get_noc_addr(((jt * 32 + q) * MT + bt) * Kt + n) + half * HALF, stage + q * HALF, HALF);
                noc_async_read_barrier();
                for (uint32_t k = 0; k < 16; ++k) {
                    const uint32_t tile = dst + (k * Kt + n) * tb;
                    for (uint32_t q = 0; q < 32; ++q) {
                        const uint32_t src = stage + q * HALF + k * ROW;
                        noc_async_read(get_noc_addr(src), tile + row(q), ROW);
                        noc_async_read(get_noc_addr(src + FACE), tile + row(q) + FACE, ROW);
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(x_cb, 16 * Kt);
        }
        return;
    }
    uint32_t page = r0 * Kt;
    for (uint32_t i = 0; i < nrows; i += RB) {
        const uint32_t n = (nrows - i < RB) ? (nrows - i) : RB;
        cb_reserve_back(x_cb, n * Kt);
        uint32_t dst = get_write_ptr(x_cb);
        for (uint32_t t = 0; t < n * Kt; ++t, ++page, dst += tb) noc_async_read_page(page, x, dst);
        noc_async_read_barrier();
        cb_push_back(x_cb, n * Kt);
    }
}
