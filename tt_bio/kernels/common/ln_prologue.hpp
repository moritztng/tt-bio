// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The channel LayerNorm prologue of the resident pair kernels (trimul T1 and T2): one 32-row tile of
// x, Kt channel tiles, normalised with two-pass statistics in fp32 CBs (mean, then the biased
// variance of x - mean, as ttnn.layer_norm), times gamma plus beta broadcast from row 0.
//
// C names the circular buffers and Kt as static constexpr members (x, gamma, beta, scaler, eps,
// mean, var, rstd, xc, tmp). The last stage, + beta, leaves channel tile i in DST 0 and calls
// pack(i), so each kernel packs where its consumer reads: T1 into a reserved slot of its unit,
// T2 appended to its GEMM input. The pack hook does no CB bookkeeping; the caller reserves and
// pushes its output. Needs REDUCE_OP / REDUCE_DIM from the host (SUM, REDUCE_ROW) and the scaler
// tile holding 1/K.
#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/rsqrt.h"

namespace ln_prologue {

// Sum of each row of the Kt tiles at the front of `in`, times the 1/K scaler, into one tile.
template <class C>
ALWI void row_mean(uint32_t in, uint32_t out, bool pop) {
    reconfig_data_format(in, C::scaler);
    pack_reconfig_data_format(out);
    cb_wait_front(in, C::Kt);
    cb_reserve_back(out, 1);
    tile_regs_acquire();
    reduce_init<REDUCE_OP, REDUCE_DIM, true>(in, C::scaler, out);
    for (uint32_t i = 0; i < C::Kt; ++i) reduce_tile<REDUCE_OP, REDUCE_DIM, true>(in, C::scaler, i, 0, 0);
    reduce_uninit<true>();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    if (pop) cb_pop_front(in, C::Kt);
}

enum Op { SUB_COL, MUL_COL, MUL_ROW, MUL };

// out[i] = a[i] (op) b[i or 0], Kt tiles. Neither input is popped here.
template <class C, Op OP>
ALWI void stage(uint32_t a, uint32_t b, uint32_t out) {
    reconfig_data_format(a, b);
    pack_reconfig_data_format(out);
    if constexpr (OP == SUB_COL) sub_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_COL) mul_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_ROW) mul_bcast_rows_init_short(a, b);
    if constexpr (OP == MUL) mul_tiles_init(a, b);
    cb_reserve_back(out, C::Kt);
    for (uint32_t i = 0; i < C::Kt; ++i) {
        tile_regs_acquire();
        if constexpr (OP == SUB_COL) sub_tiles_bcast_cols(a, b, i, 0, 0);
        if constexpr (OP == MUL_COL) mul_tiles_bcast_cols(a, b, i, 0, 0);
        if constexpr (OP == MUL_ROW) mul_tiles_bcast_rows(a, b, i, i, 0);
        if constexpr (OP == MUL) mul_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out);
        tile_regs_release();
    }
    cb_push_back(out, C::Kt);
}

// Kt tiles at the front of C::x (popped) -> pack(i) for every channel tile i, `out` the CB pack
// writes to (its data format is configured here).
template <class C, class Pack>
ALWI void layer_norm_row(uint32_t out, Pack pack) {
    row_mean<C>(C::x, C::mean, false);                    // mean
    cb_wait_front(C::mean, 1);
    stage<C, SUB_COL>(C::x, C::mean, C::xc);              // xc = x - mean
    cb_pop_front(C::x, C::Kt);
    cb_pop_front(C::mean, 1);
    cb_wait_front(C::xc, C::Kt);
    stage<C, MUL>(C::xc, C::xc, C::tmp);                  // xc^2
    row_mean<C>(C::tmp, C::var, true);                    // var (biased, as layer_norm)
    cb_wait_front(C::var, 1);
    reconfig_data_format(C::var, C::eps);
    pack_reconfig_data_format(C::rstd);
    add_tiles_init(C::var, C::eps);
    cb_reserve_back(C::rstd, 1);
    tile_regs_acquire();
    add_tiles(C::var, C::eps, 0, 0, 0);                   // rstd = rsqrt(var + eps)
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, C::rstd);
    tile_regs_release();
    cb_push_back(C::rstd, 1);
    cb_pop_front(C::var, 1);
    cb_wait_front(C::rstd, 1);
    stage<C, MUL_COL>(C::xc, C::rstd, C::tmp);            // norm = xc * rstd
    cb_pop_front(C::xc, C::Kt);
    cb_pop_front(C::rstd, 1);
    cb_wait_front(C::tmp, C::Kt);
    stage<C, MUL_ROW>(C::tmp, C::gamma, C::xc);           // * gamma
    cb_pop_front(C::tmp, C::Kt);
    cb_wait_front(C::xc, C::Kt);
    reconfig_data_format(C::xc, C::beta);                 // + beta, packed by the caller's hook
    pack_reconfig_data_format(out);
    add_bcast_rows_init_short(C::xc, C::beta);
    for (uint32_t i = 0; i < C::Kt; ++i) {
        tile_regs_acquire();
        add_tiles_bcast_rows(C::xc, C::beta, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack(i);
        tile_regs_release();
    }
    cb_pop_front(C::xc, C::Kt);
}

}  // namespace ln_prologue
