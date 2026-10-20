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
// One launch covers the channel group reader.cpp names (Q0, QN of CT2; NG g_out tiles) and IL row
// tiles per unit (32, or 16 where c_z 256 would not fit L1); every "32" above is IL then.
// fp32 DEST (the statistics), the caller's fidelity. Step 1 is kernels/common/ln_prologue.hpp.
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
#if defined(ABL) && (ABL & 384)
#define LN_ABL ((ABL >> 7) & 3)   // bench-only LN stage skips, see ln_prologue.hpp
#endif
#include "../common/ln_prologue.hpp"
// ABL (bench-only define, trimul_tail.T1_ABL; 0 in every model path): bits that skip one stage's
// work while keeping its circular-buffer traffic, to time each stage by difference (perf/spd_trimul/in_t1.py --abl).
//   1 layer_norm, 2 g_out matmul, 4 [a | b] matmuls, 8 gate multiply, 16 z reads, 32 writer gather, 64 writer writes,
//   128 the layer_norm's gamma / beta stages, 256 its variance square (ln_prologue.hpp LN_ABL).
#ifndef ABL
#define ABL 0
#endif
// T1XOUT / T1PRE (trimul_tail.T1_LN1, the c_z 256 / 384 plans' channel groups): the first launch also packs
// every normalised tile to c_20 for the writer, which stores xn in z's layout; the later launches skip the
// layer_norm and take xn from the reader straight into xn_cb. The same bf16 tiles feed every matmul either way.
#ifndef T1XOUT
#define T1XOUT 0
#endif
#ifndef T1PRE
#define T1PRE 0
#endif

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t QN = get_compile_time_arg_val(1);        // [a | b] channel tiles of this launch
constexpr uint32_t SBW = get_compile_time_arg_val(2);
constexpr uint32_t MASK = get_compile_time_arg_val(3);
constexpr uint32_t SIGPOLY = get_compile_time_arg_val(4);   // 1/2: sigmoid_poly.hpp's degree-8/6 polynomial
// Round to nearest even before a pack (the pack truncates): bit 0 the output, bit 1 p and g.
constexpr uint32_t RNE = get_compile_time_arg_val(5);
constexpr uint32_t NG = get_compile_time_arg_val(6);        // g_out channel tiles of this launch, 0: none
constexpr uint32_t Q0 = get_compile_time_arg_val(7);        // first [a | b] channel tile of this launch
constexpr uint32_t CT2 = get_compile_time_arg_val(8);       // [a | b] channel tiles in all; q < CT2 / 2 is a
constexpr uint32_t IL = get_compile_time_arg_val(9);        // row tiles per unit, 32 or 16 (reader.cpp)
constexpr uint32_t GSBW = NG % 4 == 0 ? 4 : (NG % 2 == 0 ? 2 : 1);
static_assert(IL % SBW == 0 && IL % 4 == 0 && SBW <= 4, "fp32 half-sync DST holds 4 tiles");

constexpr uint32_t w_cb = tt::CBIndex::c_0;      // [Wp^T | Wg^T] of the group, 2 * QN * Kt, resident
constexpr uint32_t xn_cb = tt::CBIndex::c_1;     // layer_norm(z) of the unit, bf16, IL * Kt
constexpr uint32_t out_cb = tt::CBIndex::c_2;    // a / b tiles to the writer
constexpr uint32_t p_cb = tt::CBIndex::c_4;
constexpr uint32_t g_cb = tt::CBIndex::c_5;
constexpr uint32_t m_cb = tt::CBIndex::c_6;
// g * m of one 4-tile block. The mask goes into g through its own CB and p * (g m) is a plain binary op:
// reusing DEST for p * (g m) left every unit but each core's last wrong on Wormhole at c_z 256 and 384,
// the unit's tail racing the next unit's layer_norm.
constexpr uint32_t gm_cb = tt::CBIndex::c_15;
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
constexpr uint32_t wo_cb = tt::CBIndex::c_18;    // Wg_out [Kt x NG] of the group, resident
constexpr uint32_t go_cb = tt::CBIndex::c_19;    // g_out of one row tile, NG tiles
constexpr uint32_t xo_cb = tt::CBIndex::c_20;    // T1XOUT: one row tile of xn to the writer, Kt tiles

ALWI void sigmoid_bf16_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid<false, false>(idst, (int)VectorMode::RC)));
}

struct LnCbs {
    static constexpr uint32_t Kt = ::Kt, x = cb_x, gamma = cb_gamma, beta = cb_beta, scaler = cb_scaler,
        eps = cb_eps, mean = cb_mean, var = cb_var, rstd = cb_rstd, xc = cb_xc, tmp = cb_tmp;
};

// Row tile il: Kt tiles at the front of cb_x (popped) -> slots dt * IL + il of the reserved xn_cb.
ALWI void layer_norm_row(uint32_t il) {
    if constexpr (ABL & 1) {
        cb_wait_front(cb_x, Kt);
        cb_pop_front(cb_x, Kt);
        if constexpr (T1XOUT) {
            cb_reserve_back(xo_cb, Kt);
            cb_push_back(xo_cb, Kt);
        }
        return;
    }
    if constexpr (T1XOUT) {
        cb_reserve_back(xo_cb, Kt);
        ln_prologue::layer_norm_row<LnCbs>(xn_cb, [il](uint32_t i) {
            pack_tile<true>(0, xn_cb, i * IL + il);
            pack_tile<true>(0, xo_cb, i);
        });
        cb_push_back(xo_cb, Kt);
    } else {
        ln_prologue::layer_norm_row<LnCbs>(xn_cb, [il](uint32_t i) { pack_tile<true>(0, xn_cb, i * IL + il); });
    }
}

void kernel_main() {
    const uint32_t nunits = get_arg_val<uint32_t>(0);

    mm_init(w_cb, xn_cb, p_cb, 1);
    cb_wait_front(w_cb, 2 * QN * Kt);
    cb_wait_front(cb_gamma, Kt);
    cb_wait_front(cb_beta, Kt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    if constexpr (NG) cb_wait_front(wo_cb, Kt * NG);

    for (uint32_t u = 0; u < nunits; ++u) {
        // 1. layer_norm of the unit's IL row tiles (T1PRE: the reader fills xn_cb with them).
        if constexpr (!T1PRE) {
            cb_reserve_back(xn_cb, IL * Kt);
            for (uint32_t il = 0; il < IL; ++il) layer_norm_row(il);
            cb_push_back(xn_cb, IL * Kt);
        }
        cb_wait_front(xn_cb, IL * Kt);

        // 2. g_out = xn @ Wg_out, row tile by row tile, one K block in k order.
        if constexpr (NG) {
            mm_block_init_short(xn_cb, wo_cb, false, GSBW, 1, Kt);
            reconfig_data_format(wo_cb, xn_cb);
            pack_reconfig_data_format(go_cb);
            for (uint32_t il = 0; il < IL; ++il) {
                cb_reserve_back(go_cb, NG);
                for (uint32_t n0 = 0; n0 < NG && !(ABL & 2); n0 += GSBW) {
                    tile_regs_acquire();
                    uint32_t i0 = il;
                    uint32_t i1 = n0;
                    for (uint32_t k = 0; k < Kt; ++k, i0 += IL, i1 += NG)
                        matmul_block(xn_cb, wo_cb, i0, i1, 0, false, GSBW, 1, Kt);
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < GSBW; ++i) pack_tile<true>(i, go_cb, n0 + i);
                    tile_regs_release();
                }
                cb_push_back(go_cb, NG);
            }
        }

        // 3. gin_moved's gated projection on xn. The sigmoid is re-initialised every unit: the
        // prologue's rsqrt init reprograms the SFPU (as in T2).
        sigmoid_tile_init();
        if constexpr (MASK) cb_wait_front(m_cb, IL);
        for (uint32_t q = 0; q < QN; ++q) {
            for (uint32_t pass = 0; pass < 2; ++pass) {
                const uint32_t pass_cb = pass == 0 ? p_cb : g_cb;
                mm_block_init_short(w_cb, xn_cb, true, SBW, 1, Kt);
                reconfig_data_format(xn_cb, w_cb);
                pack_reconfig_data_format(pass_cb);
                cb_reserve_back(pass_cb, IL);
                for (uint32_t il0 = 0; il0 < IL && !(ABL & 4); il0 += SBW) {
                    tile_regs_acquire();
                    uint32_t i0 = (pass * QN + q) * Kt;
                    uint32_t i1 = il0;
                    for (uint32_t k = 0; k < Kt; ++k, ++i0, i1 += IL)
                        matmul_block(w_cb, xn_cb, i0, i1, 0, true, SBW, 1, Kt);
                    if (pass == 1) {
                        for (uint32_t i = 0; i < SBW; ++i) {
                            if constexpr (SIGPOLY) sigmoid_poly_tile<SIGPOLY>(i);
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
                cb_push_back(pass_cb, IL);
            }
            cb_wait_front(p_cb, IL);
            cb_wait_front(g_cb, IL);
            cb_reserve_back(out_cb, IL);
            reconfig_data_format(p_cb, g_cb);
            pack_reconfig_data_format(out_cb);
            for (uint32_t t0 = 0; t0 < IL && !(ABL & 8); t0 += 4) {
                const bool masked = MASK && Q0 + q < CT2 / 2;
                if (masked) {
                    cb_reserve_back(gm_cb, 4);
                    tile_regs_acquire();
                    mul_bcast_rows_init_short(g_cb, m_cb);
                    for (uint32_t i = 0; i < 4; ++i) mul_tiles_bcast_rows(g_cb, m_cb, t0 + i, t0 + i, i);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(gm_cb);
                    for (uint32_t i = 0; i < 4; ++i) pack_tile(i, gm_cb);
                    tile_regs_release();
                    cb_push_back(gm_cb, 4);
                    cb_wait_front(gm_cb, 4);
                    pack_reconfig_data_format(out_cb);
                }
                tile_regs_acquire();
                if (masked) {
                    mul_tiles_init(p_cb, gm_cb);
                    for (uint32_t i = 0; i < 4; ++i) mul_tiles(p_cb, gm_cb, t0 + i, i, i);
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
                if (masked) cb_pop_front(gm_cb, 4);
            }
            cb_push_back(out_cb, IL);
            cb_pop_front(p_cb, IL);
            cb_pop_front(g_cb, IL);
        }
        cb_pop_front(xn_cb, IL * Kt);
        if constexpr (MASK) cb_pop_front(m_cb, IL);
    }
}
