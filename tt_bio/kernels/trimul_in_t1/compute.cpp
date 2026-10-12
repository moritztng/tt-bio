// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul input side T1, weights resident. Per unit (b, xt, yt), 32 row tiles R_il of the raw pair z:
//   1. layer_norm, row tile by row tile: T2's prologue (two-pass statistics in fp32 CBs, scaler 1/K,
//      * gamma + beta broadcast from row 0), the result packed bf16 into c_1 at slot dt * 32 + il,
//      the slot order gin_moved's reader gives its activation, so step 3 is gin_moved's loop as is.
//   2. g_out = xn @ Wg_out per row tile (in1 = Wg_out, untransposed), packed to c_19 in z's layout.
//   3. per output channel tile q of [a | b]: 32 tiles W^T[q] @ xn_il^T (value pass, gate pass with
//      the sigmoid in DST), then a = p * sigmoid(g) [* m] packed in il order for the writer, which
//      gathers them into the [B, C, S, S] tiles the einsum reads.
// fp32 DEST (the statistics), the caller's fidelity. NEVER RUN ON A DEVICE YET.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "../trimul_gin_moved/sigmoid_poly.hpp"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t CT2 = get_compile_time_arg_val(1);
constexpr uint32_t SBW = get_compile_time_arg_val(2);
constexpr uint32_t MASK = get_compile_time_arg_val(3);
constexpr uint32_t SIGPOLY = get_compile_time_arg_val(4);   // 1: sigmoid_poly.hpp's polynomial
// Round to nearest even before a pack (the pack truncates): bit 0 the output, bit 1 p and g.
constexpr uint32_t RNE = get_compile_time_arg_val(5);
constexpr uint32_t NTG = get_compile_time_arg_val(6);       // g_out channel tiles, 0: none
constexpr uint32_t GSBW = NTG % 4 == 0 ? 4 : (NTG % 2 == 0 ? 2 : 1);
static_assert(32 % SBW == 0 && SBW <= 4, "fp32 half-sync DST holds 4 tiles");

constexpr uint32_t w_cb = tt::CBIndex::c_0;      // [Wp^T | Wg^T], 2 * CT2 * Kt, resident
constexpr uint32_t xn_cb = tt::CBIndex::c_1;     // layer_norm(z) of the unit, bf16, 32 * Kt
constexpr uint32_t out_cb = tt::CBIndex::c_2;    // a / b tiles to the writer
constexpr uint32_t p_cb = tt::CBIndex::c_4;
constexpr uint32_t g_cb = tt::CBIndex::c_5;
constexpr uint32_t m_cb = tt::CBIndex::c_6;
constexpr uint32_t cb_x = tt::CBIndex::c_7;      // raw z rows, Kt per row tile
constexpr uint32_t cb_gamma = tt::CBIndex::c_8;
constexpr uint32_t cb_beta = tt::CBIndex::c_9;
constexpr uint32_t cb_scaler = tt::CBIndex::c_10;
constexpr uint32_t cb_eps = tt::CBIndex::c_11;   // fp32, every element eps
constexpr uint32_t cb_mean = tt::CBIndex::c_12;  // fp32, one tile (column 0)
constexpr uint32_t cb_var = tt::CBIndex::c_13;
constexpr uint32_t cb_rstd = tt::CBIndex::c_14;
constexpr uint32_t cb_xc = tt::CBIndex::c_16;    // fp32, Kt tiles
constexpr uint32_t cb_tmp = tt::CBIndex::c_17;   // fp32, Kt tiles
constexpr uint32_t wo_cb = tt::CBIndex::c_18;    // Wg_out [Kt x NTG], resident
constexpr uint32_t go_cb = tt::CBIndex::c_19;    // g_out of one row tile, NTG tiles

ALWI void sigmoid_bf16_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid<false, false>(idst, (int)VectorMode::RC)));
}

// Sum of each row of the Kt tiles at the front of `in`, times the 1/K scaler, into one tile.
ALWI void row_mean(uint32_t in, uint32_t out, bool pop) {
    reconfig_data_format(in, cb_scaler);
    pack_reconfig_data_format(out);
    cb_wait_front(in, Kt);
    cb_reserve_back(out, 1);
    tile_regs_acquire();
    reduce_init<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, out);
    for (uint32_t i = 0; i < Kt; ++i) reduce_tile<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, i, 0, 0);
    reduce_uninit<true>();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    if (pop) cb_pop_front(in, Kt);
}

enum Op { SUB_COL, MUL_COL, MUL_ROW, MUL };

// out[i] = a[i] (op) b[i or 0], Kt tiles. Neither input is popped here.
template <Op OP>
ALWI void stage(uint32_t a, uint32_t b, uint32_t out) {
    reconfig_data_format(a, b);
    pack_reconfig_data_format(out);
    if constexpr (OP == SUB_COL) sub_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_COL) mul_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_ROW) mul_bcast_rows_init_short(a, b);
    if constexpr (OP == MUL) mul_tiles_init(a, b);
    cb_reserve_back(out, Kt);
    for (uint32_t i = 0; i < Kt; ++i) {
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
    cb_push_back(out, Kt);
}

// Row tile il: Kt tiles at the front of cb_x (popped) -> slots dt * 32 + il of the reserved xn_cb.
ALWI void layer_norm_row(uint32_t il) {
    row_mean(cb_x, cb_mean, false);                       // mean
    cb_wait_front(cb_mean, 1);
    stage<SUB_COL>(cb_x, cb_mean, cb_xc);                 // xc = x - mean
    cb_pop_front(cb_x, Kt);
    cb_pop_front(cb_mean, 1);
    cb_wait_front(cb_xc, Kt);
    stage<MUL>(cb_xc, cb_xc, cb_tmp);                     // xc^2
    row_mean(cb_tmp, cb_var, true);                       // var (biased, as layer_norm)
    cb_wait_front(cb_var, 1);
    reconfig_data_format(cb_var, cb_eps);
    pack_reconfig_data_format(cb_rstd);
    add_tiles_init(cb_var, cb_eps);
    cb_reserve_back(cb_rstd, 1);
    tile_regs_acquire();
    add_tiles(cb_var, cb_eps, 0, 0, 0);                   // rstd = rsqrt(var + eps)
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rstd);
    tile_regs_release();
    cb_push_back(cb_rstd, 1);
    cb_pop_front(cb_var, 1);
    cb_wait_front(cb_rstd, 1);
    stage<MUL_COL>(cb_xc, cb_rstd, cb_tmp);               // norm = xc * rstd
    cb_pop_front(cb_xc, Kt);
    cb_pop_front(cb_rstd, 1);
    cb_wait_front(cb_tmp, Kt);
    stage<MUL_ROW>(cb_tmp, cb_gamma, cb_xc);              // * gamma
    cb_pop_front(cb_tmp, Kt);
    cb_wait_front(cb_xc, Kt);
    // + beta, packed bf16 straight into the unit's slot for (dt, il); xn_cb is reserved by the caller.
    reconfig_data_format(cb_xc, cb_beta);
    pack_reconfig_data_format(xn_cb);
    add_bcast_rows_init_short(cb_xc, cb_beta);
    for (uint32_t i = 0; i < Kt; ++i) {
        tile_regs_acquire();
        add_tiles_bcast_rows(cb_xc, cb_beta, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, xn_cb, i * 32 + il);
        tile_regs_release();
    }
    cb_pop_front(cb_xc, Kt);
}

void kernel_main() {
    const uint32_t nunits = get_arg_val<uint32_t>(0);

    mm_init(w_cb, xn_cb, p_cb, 1);
    cb_wait_front(w_cb, 2 * CT2 * Kt);
    cb_wait_front(cb_gamma, Kt);
    cb_wait_front(cb_beta, Kt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    if constexpr (NTG) cb_wait_front(wo_cb, Kt * NTG);

    for (uint32_t u = 0; u < nunits; ++u) {
        // 1. layer_norm of the unit's 32 row tiles.
        cb_reserve_back(xn_cb, 32 * Kt);
        for (uint32_t il = 0; il < 32; ++il) layer_norm_row(il);
        cb_push_back(xn_cb, 32 * Kt);
        cb_wait_front(xn_cb, 32 * Kt);

        // 2. g_out = xn @ Wg_out, row tile by row tile, one K block in k order.
        if constexpr (NTG) {
            mm_block_init_short(xn_cb, wo_cb, false, GSBW, 1, Kt);
            reconfig_data_format(wo_cb, xn_cb);
            pack_reconfig_data_format(go_cb);
            for (uint32_t il = 0; il < 32; ++il) {
                cb_reserve_back(go_cb, NTG);
                for (uint32_t n0 = 0; n0 < NTG; n0 += GSBW) {
                    tile_regs_acquire();
                    uint32_t i0 = il;
                    uint32_t i1 = n0;
                    for (uint32_t k = 0; k < Kt; ++k, i0 += 32, i1 += NTG)
                        matmul_block(xn_cb, wo_cb, i0, i1, 0, false, GSBW, 1, Kt);
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < GSBW; ++i) pack_tile<true>(i, go_cb, n0 + i);
                    tile_regs_release();
                }
                cb_push_back(go_cb, NTG);
            }
        }

        // 3. gin_moved's gated projection on xn. The sigmoid is re-initialised every unit: the
        // prologue's rsqrt init reprograms the SFPU (as in T2).
        sigmoid_tile_init();
        if constexpr (MASK) cb_wait_front(m_cb, 32);
        for (uint32_t q = 0; q < CT2; ++q) {
            for (uint32_t pass = 0; pass < 2; ++pass) {
                const uint32_t pass_cb = pass == 0 ? p_cb : g_cb;
                mm_block_init_short(w_cb, xn_cb, true, SBW, 1, Kt);
                reconfig_data_format(xn_cb, w_cb);
                pack_reconfig_data_format(pass_cb);
                cb_reserve_back(pass_cb, 32);
                for (uint32_t il0 = 0; il0 < 32; il0 += SBW) {
                    tile_regs_acquire();
                    uint32_t i0 = (pass * CT2 + q) * Kt;
                    uint32_t i1 = il0;
                    for (uint32_t k = 0; k < Kt; ++k, ++i0, i1 += 32)
                        matmul_block(w_cb, xn_cb, i0, i1, 0, true, SBW, 1, Kt);
                    if (pass == 1) {
                        for (uint32_t i = 0; i < SBW; ++i) {
                            if constexpr (SIGPOLY) sigmoid_poly_tile(i);
                            else sigmoid_bf16_tile(i);
                        }
                    }
                    if constexpr (RNE & 2) {
                        for (uint32_t i = 0; i < SBW; ++i) round_bf16_rne_tile(i);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < SBW; ++i) pack_tile<true>(i, pass_cb, il0 + i);
                    tile_regs_release();
                }
                cb_push_back(pass_cb, 32);
            }
            cb_wait_front(p_cb, 32);
            cb_wait_front(g_cb, 32);
            cb_reserve_back(out_cb, 32);
            reconfig_data_format(p_cb, g_cb);
            pack_reconfig_data_format(out_cb);
            for (uint32_t t0 = 0; t0 < 32; t0 += 4) {
                tile_regs_acquire();
                if (MASK && q < CT2 / 2) {
                    mul_bcast_rows_init_short(g_cb, m_cb);
                    for (uint32_t i = 0; i < 4; ++i) mul_tiles_bcast_rows(g_cb, m_cb, t0 + i, t0 + i, i);
                    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        p_cb);
                    for (uint32_t i = 0; i < 4; ++i)
                        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                            p_cb, t0 + i, i);
                } else {
                    mul_tiles_init(p_cb, g_cb);
                    for (uint32_t i = 0; i < 4; ++i) mul_tiles(p_cb, g_cb, t0 + i, t0 + i, i);
                }
                if constexpr (RNE & 1) {
                    for (uint32_t i = 0; i < 4; ++i) round_bf16_rne_tile(i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < 4; ++i) pack_tile(i, out_cb);
                tile_regs_release();
            }
            cb_push_back(out_cb, 32);
            cb_pop_front(p_cb, 32);
            cb_pop_front(g_cb, 32);
        }
        cb_pop_front(xn_cb, 32 * Kt);
        if constexpr (MASK) cb_pop_front(m_cb, 32);
    }
}
