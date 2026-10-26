// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The channel LayerNorm prologue of the resident pair kernels (trimul T1 and T2, pair_swiglu_ws; the
// RMSNorm form for pair_swiglu_ws under WS_RMS): one 32-row tile of
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
#include "api/compute/tile_move_copy.h"

// LN_ABL (bench-only, 0 in every model path; trimul_in_t1 maps ABL bits 128 / 256 to it): bit 0 skips the * gamma and
// + beta stages (the normalised tile is copied to the pack hook), bit 1 reduces x - mean directly instead of its
// square. Each prices what fusing those stages away could save.
#ifndef LN_ABL
#define LN_ABL 0
#endif
// LN_BLK: channel tiles per dest acquire in layer_norm_row's eltwise stages (1 or 4, a half-sync fp32 dest holds 4).
// Same ops on the same values in other dest slots, so bit-identical; a kernel opts in with the define.
#ifndef LN_BLK
#define LN_BLK 1
#endif

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

// out[i] = a[i] (op) b[i or 0], Kt tiles, BLK of them per dest acquire (4 fits a half-sync fp32 dest). Neither input
// is popped here.
template <class C, Op OP, uint32_t BLK = 1>
ALWI void stage(uint32_t a, uint32_t b, uint32_t out) {
    reconfig_data_format(a, b);
    pack_reconfig_data_format(out);
    if constexpr (OP == SUB_COL) sub_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_COL) mul_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_ROW) mul_bcast_rows_init_short(a, b);
    if constexpr (OP == MUL) mul_tiles_init(a, b);
    cb_reserve_back(out, C::Kt);
    for (uint32_t i0 = 0; i0 < C::Kt; i0 += BLK) {
        const uint32_t n = C::Kt - i0 < BLK ? C::Kt - i0 : BLK;
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t i = i0 + j;
            if constexpr (OP == SUB_COL) sub_tiles_bcast_cols(a, b, i, 0, j);
            if constexpr (OP == MUL_COL) mul_tiles_bcast_cols(a, b, i, 0, j);
            if constexpr (OP == MUL_ROW) mul_tiles_bcast_rows(a, b, i, i, j);
            if constexpr (OP == MUL) mul_tiles(a, b, i, i, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) pack_tile(j, out);
        tile_regs_release();
    }
    cb_push_back(out, C::Kt);
}

// rstd = rsqrt(var + eps) from the one tile at the front of C::var (popped) into C::rstd (waited on).
template <class C>
ALWI void rstd() {
    cb_wait_front(C::var, 1);
    reconfig_data_format(C::var, C::eps);
    pack_reconfig_data_format(C::rstd);
    add_tiles_init(C::var, C::eps);
    cb_reserve_back(C::rstd, 1);
    tile_regs_acquire();
    add_tiles(C::var, C::eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, C::rstd);
    tile_regs_release();
    cb_push_back(C::rstd, 1);
    cb_pop_front(C::var, 1);
    cb_wait_front(C::rstd, 1);
}

// Kt tiles at the front of C::x (popped) -> pack(i) for every channel tile i, `out` the CB pack
// writes to (its data format is configured here).
template <class C, class Pack>
ALWI void layer_norm_row(uint32_t out, Pack pack) {
    row_mean<C>(C::x, C::mean, false);                    // mean
    cb_wait_front(C::mean, 1);
    stage<C, SUB_COL, LN_BLK>(C::x, C::mean, C::xc);              // xc = x - mean
    cb_pop_front(C::x, C::Kt);
    cb_pop_front(C::mean, 1);
    cb_wait_front(C::xc, C::Kt);
    if constexpr (LN_ABL & 2) {
        row_mean<C>(C::xc, C::var, false);
    } else {
        stage<C, MUL, LN_BLK>(C::xc, C::xc, C::tmp);              // xc^2
        row_mean<C>(C::tmp, C::var, true);                // var (biased, as layer_norm)
    }
    rstd<C>();                                            // rstd = rsqrt(var + eps)
    stage<C, MUL_COL, LN_BLK>(C::xc, C::rstd, C::tmp);            // norm = xc * rstd
    cb_pop_front(C::xc, C::Kt);
    cb_pop_front(C::rstd, 1);
    cb_wait_front(C::tmp, C::Kt);
    if constexpr (LN_ABL & 1) {
        reconfig_data_format_srca(C::tmp);
        pack_reconfig_data_format(out);
        copy_tile_to_dst_init_short(C::tmp);
        for (uint32_t i = 0; i < C::Kt; ++i) {
            tile_regs_acquire();
            copy_tile(C::tmp, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack(i);
            tile_regs_release();
        }
        cb_pop_front(C::tmp, C::Kt);
        return;
    }
    stage<C, MUL_ROW, LN_BLK>(C::tmp, C::gamma, C::xc);           // * gamma
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

// RMSNorm, as ttnn.rms_norm: x * rsqrt(mean(x^2) + eps) * gamma, no centring and no beta. Same CBs
// and contract as layer_norm_row; C::beta and C::xc are not touched.
template <class C, class Pack>
ALWI void rms_norm_row(uint32_t out, Pack pack) {
    cb_wait_front(C::x, C::Kt);
    stage<C, MUL>(C::x, C::x, C::tmp);                    // x^2
    row_mean<C>(C::tmp, C::var, true);                    // mean(x^2)
    rstd<C>();
    stage<C, MUL_COL>(C::x, C::rstd, C::tmp);             // x * rstd
    cb_pop_front(C::x, C::Kt);
    cb_pop_front(C::rstd, 1);
    cb_wait_front(C::tmp, C::Kt);
    reconfig_data_format(C::tmp, C::gamma);               // * gamma, packed by the caller's hook
    pack_reconfig_data_format(out);
    mul_bcast_rows_init_short(C::tmp, C::gamma);
    for (uint32_t i = 0; i < C::Kt; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(C::tmp, C::gamma, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack(i);
        tile_regs_release();
    }
    cb_pop_front(C::tmp, C::Kt);
}

}  // namespace ln_prologue
