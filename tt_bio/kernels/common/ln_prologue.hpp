// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The channel LayerNorm prologue of the resident pair kernels (trimul T1 and T2, pair_swiglu_ws; the
// RMSNorm form for pair_swiglu_ws under WS_RMS): one 32-row tile of
// x, Kt channel tiles, normalised with two-pass statistics in fp32 CBs (mean, then the biased
// variance of x - mean, as ttnn.layer_norm), times gamma plus beta broadcast from row 0.
//
// C names the circular buffers and Kt as static constexpr members (x, gamma, beta, scaler, eps,
// mean, var, rstd, xc, tmp). The last stage, + beta, leaves channel tile i in DST d and calls
// pack(d, i), so each kernel packs where its consumer reads: T1 into a reserved slot of its unit,
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
#include "api/compute/reconfig_data_format.h"

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

// Dest-reuse binary op: DST[idst] = DST[idst] (op) b, b tile itile of cb broadcast as B (COL: column 0,
// ROW: row 0). DST moves to srcA, b unpacks to srcB. The tt-metal API only ships the NONE broadcast.
template <EltwiseBinaryType OP, BroadcastType B>
ALWI void reuse_init(uint32_t cb) {
    reconfig_data_format_srcb(cb);
    UNPACK((llk_unpack_A_init<B, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(false, false, cb)));
    MATH((llk_math_eltwise_binary_init<OP, B, OP == ELWMUL ? MATH_FIDELITY : MathFidelity::LoFi,
                                       EltwiseBinaryReuseDestType::DEST_TO_SRCA>(false)));
}

template <EltwiseBinaryType OP, BroadcastType B>
ALWI void reuse(uint32_t cb, uint32_t itile, uint32_t idst) {
    UNPACK((llk_unpack_A<B, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb, itile)));
    MATH((llk_math_eltwise_binary<OP, B, DST_ACCUM_MODE, OP == ELWMUL ? MATH_FIDELITY : MathFidelity::LoFi,
                                  EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb, cb, idst, true)));
}

// Channel tiles per DST round: fp32 half-sync DEST holds 4.
constexpr uint32_t DB = 4;

// Kt tiles at the front of C::x (popped) -> pack(d, i) for every channel tile i, held in DST d, `out`
// the CB pack writes to (its data format is configured here). Two DST rounds per DB channel tiles:
// (x - mean)^2 (squared on the SFPU in fp32) for the variance, then ((x - mean) * rstd) * gamma + beta
// with the centring recomputed from x, so no centred or normalised tile goes through a CB. C::xc is
// not touched.
template <class C, class Pack>
ALWI void layer_norm_row(uint32_t out, Pack pack) {
    row_mean<C>(C::x, C::mean, false);                    // mean
    cb_wait_front(C::mean, 1);
    pack_reconfig_data_format(C::tmp);                    // (x - mean)^2 -> C::tmp
    cb_reserve_back(C::tmp, C::Kt);
    for (uint32_t i0 = 0; i0 < C::Kt; i0 += DB) {
        const uint32_t n = C::Kt - i0 < DB ? C::Kt - i0 : DB;
        tile_regs_acquire();
        reconfig_data_format(C::x, C::mean);
        sub_bcast_cols_init_short(C::x, C::mean);
        for (uint32_t j = 0; j < n; ++j) sub_tiles_bcast_cols(C::x, C::mean, i0 + j, 0, j);
        square_tile_init();
        for (uint32_t j = 0; j < n; ++j) square_tile(j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) pack_tile(j, C::tmp);
        tile_regs_release();
    }
    cb_push_back(C::tmp, C::Kt);
    row_mean<C>(C::tmp, C::var, true);                    // var (biased, as layer_norm)
    rstd<C>();                                            // rstd = rsqrt(var + eps)
    pack_reconfig_data_format(out);
    for (uint32_t i0 = 0; i0 < C::Kt; i0 += DB) {
        const uint32_t n = C::Kt - i0 < DB ? C::Kt - i0 : DB;
        tile_regs_acquire();
        reconfig_data_format(C::x, C::mean);
        sub_bcast_cols_init_short(C::x, C::mean);
        for (uint32_t j = 0; j < n; ++j) sub_tiles_bcast_cols(C::x, C::mean, i0 + j, 0, j);
        reconfig_data_format_srca(C::tmp);                // DST (fp32) is srcA from here on
        reuse_init<ELWMUL, BroadcastType::COL>(C::rstd);
        for (uint32_t j = 0; j < n; ++j) reuse<ELWMUL, BroadcastType::COL>(C::rstd, 0, j);
        reuse_init<ELWMUL, BroadcastType::ROW>(C::gamma);
        for (uint32_t j = 0; j < n; ++j) reuse<ELWMUL, BroadcastType::ROW>(C::gamma, i0 + j, j);
        reuse_init<ELWADD, BroadcastType::ROW>(C::beta);
        for (uint32_t j = 0; j < n; ++j) reuse<ELWADD, BroadcastType::ROW>(C::beta, i0 + j, j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) pack(j, i0 + j);
        tile_regs_release();
    }
    cb_pop_front(C::x, C::Kt);
    cb_pop_front(C::mean, 1);
    cb_pop_front(C::rstd, 1);
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
        pack(0, i);
        tile_regs_release();
    }
    cb_pop_front(C::tmp, C::Kt);
}

}  // namespace ln_prologue
