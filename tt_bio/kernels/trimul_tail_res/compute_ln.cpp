// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail T2, weights resident: z += (layer_norm(x) @ Wp) * sigmoid(g), one Mb x Nt block at a
// time, z written in place. The four ops of the c_z = 128 tail (layer_norm, _pair_proj_linear,
// multiply_ with SIGMOID on b, add_) in one program. Each core holds whole rows of K, so the
// channel LayerNorm is local: per 32-row tile, the lnbw kernel's two-pass statistics (fp32 CBs,
// fp32 DEST reductions, scaler 1/K exact for K a power of two), then * gamma + beta (row 0 of each
// gamma/beta tile broadcast down the rows) packed to bf16 into c_3, the CB the GEMM reads. Wp sits
// in c_1 for the whole program. The GEMM packs p to bf16 (c_4) as the incumbent's L1 p_out does;
// the epilogue copies g (bf16, read from DRAM, c_5) into DST, applies the sigmoid there, multiplies
// by p and adds z (c_7) by DEST reuse, and packs to c_2 for the writer. NEVER RUN ON A DEVICE YET.
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
#include "api/compute/transpose_wh.h"
#include "../trimul_gin_moved/sigmoid_poly.hpp"
#include "../common/ln_prologue.hpp"
#include "../common/mm_fid.hpp"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Nt = get_compile_time_arg_val(1);
constexpr uint32_t Mb = get_compile_time_arg_val(2);
constexpr uint32_t SBW = get_compile_time_arg_val(3);
constexpr uint32_t SIGPOLY = get_compile_time_arg_val(4);   // 1/2: sigmoid_poly.hpp's degree-8/6 polynomial
// Round DST to bf16 nearest-even: bit 0 before the output pack, bit 2 the product before the
// residual add (as the res compute's bits 0 / 2).
constexpr uint32_t RNE = get_compile_time_arg_val(5);
// GATE 1: g is not read but computed here, sigmoid(xb @ Wg) with xb the trimul's input norm (c_6) and Wg
// resident in c_1 after Wp, as trimul_tail_res's pass 1. The callers whose in-projection does not
// leave a g_out (gin_moved, c_z 256) take this form.
constexpr uint32_t GATE = get_compile_time_arg_val(6);
// MOVED 1 (T2m, reader_moved.cpp): x arrives per row as CP tiles already in the output layout (c_18)
// and Kt - CP gathered tiles M = x^T (c_15); this kernel assembles the row in cb_x, transposing M.
constexpr uint32_t MOVED = get_compile_time_arg_val(7);
constexpr uint32_t CP = get_compile_time_arg_val(8);
// ABL (bench-only, trimul_tail.T2_ABL; 0 in every model path): bits that skip one stage's compute and
// push its CB as is, so the stage's time drops out while the reader and writer move the same bytes.
// 1 LayerNorm, 2 the p GEMM, 4 the gate GEMM + sigmoid (GATE), 8 the epilogue, 16 the epilogue's sigmoid (not GATE).
constexpr uint32_t ABL = get_compile_time_arg_val(9);
// EXTRA (MOVED only, 0 for every caller before AF2): 1 a bias on p, 2 a bias on the GATE g, each as Nt
// tiles with the bias row in all 32 rows (c_19, p first), copied into DST before the K accumulation;
// 4 the update times a per-row dropout mask (Nt tiles per block in c_20, shared by the block's rows),
// rounded with the product under RNE bit 2, as rne_add_masked rounds it. 8 is the writer's.
constexpr uint32_t EXTRA = get_compile_time_arg_val(10);
// FOLD 1 (trimul_tail.T2_FOLD, GATE and not MOVED only): the host folded the LN's centring and gamma into Wp,
// Wc = (I - 11^T / K) diag(gamma) Wp, and beta @ Wp into c_9 as Nt full bias tiles. Here only rstd is computed
// (E[x^2] - mean^2), the GEMM reads x itself, and the epilogue is rstd * (x @ Wc) + bias.
constexpr uint32_t FOLD = get_compile_time_arg_val(11);

// T2_MM_FID (trimul_tail.t2_mm_fid, lever trimul_tail_hifi2): the p GEMM's fidelity. The LayerNorm and the epilogue
// multiplies keep the compute config's, so at HiFi2 the only bit dropped is layer_norm(x)'s last one, in srcB.
#ifdef T2_MM_FID
constexpr auto MM_FID = static_cast<MathFidelity>(T2_MM_FID);
#else
constexpr auto MM_FID = static_cast<MathFidelity>(MATH_FIDELITY);
#endif

constexpr uint32_t cb_x = tt::CBIndex::c_0;      // x_out rows, bf16, Mb * Kt double buffered
constexpr uint32_t cb_w = tt::CBIndex::c_1;      // Wp, Kt * Nt, resident
constexpr uint32_t cb_out = tt::CBIndex::c_2;    // z + update, to the writer
constexpr uint32_t cb_ln = tt::CBIndex::c_3;     // layer_norm(x) rows, bf16, Mb * Kt
constexpr uint32_t cb_p = tt::CBIndex::c_4;      // x_norm @ Wp, bf16, Mb * Nt
constexpr uint32_t cb_g = tt::CBIndex::c_5;      // g_out rows, bf16, from DRAM (GATE: sigmoid(xb @ Wg))
constexpr uint32_t cb_xb = tt::CBIndex::c_6;     // GATE: xb rows, bf16, Mb * Kt double buffered
constexpr uint32_t cb_z = tt::CBIndex::c_7;      // z rows, bf16, from DRAM
constexpr uint32_t cb_gamma = tt::CBIndex::c_8;  // Kt tiles, resident
constexpr uint32_t cb_beta = tt::CBIndex::c_9;   // Kt tiles, resident
constexpr uint32_t cb_scaler = tt::CBIndex::c_10;
constexpr uint32_t cb_eps = tt::CBIndex::c_11;   // fp32, every element eps
constexpr uint32_t cb_mean = tt::CBIndex::c_12;  // fp32, one tile (column 0)
constexpr uint32_t cb_var = tt::CBIndex::c_13;
constexpr uint32_t cb_rstd = tt::CBIndex::c_14;
constexpr uint32_t cb_xc = tt::CBIndex::c_16;    // fp32, Kt tiles
constexpr uint32_t cb_tmp = tt::CBIndex::c_17;   // fp32, Kt tiles
constexpr uint32_t cb_in = tt::CBIndex::c_15;    // MOVED: gathered x^T tiles, one at a time
constexpr uint32_t cb_prev = tt::CBIndex::c_18;  // MOVED: the CP earlier channel tiles of the row
constexpr uint32_t cb_bias = tt::CBIndex::c_19;  // EXTRA 1 / 2: row-replicated bias tiles, resident
constexpr uint32_t cb_mask = tt::CBIndex::c_20;  // EXTRA 4: the block's Nt mask tiles
constexpr uint32_t NBIAS = (EXTRA & 1) + ((EXTRA >> 1) & 1);
// T2_RES32 (RFD4's float32 pair stream; not with FOLD or EXTRA): z (c_7) is float32, unpacked straight to DST
// (UnpackToDestFp32, set by the host) beside the update and added by the SFPU, so z + update is one float32
// rounding; the sum is packed float32 to c_2 (written over z) and bfloat16 to c_21 (over the trimul's input, the
// copy the next sublayer reads). Two tiles a group, the update in DST 0-1 and z in 2-3.
#ifdef T2_RES32
constexpr uint32_t cb_copy = tt::CBIndex::c_21;
constexpr uint32_t G = 2;
static_assert(!FOLD && !EXTRA, "T2_RES32 is the plain or GATE form");
#else
constexpr uint32_t G = 4;
#endif

ALWI void sigmoid_bf16_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid<false, false>(idst, (int)VectorMode::RC)));
}

// SBW bias tiles from c_19 (from tile `first`) into DST 0.., then back to the GEMM's configuration.
ALWI void bias_to_dst(uint32_t first, uint32_t cb_a) {
    reconfig_data_format_srca(cb_bias);
    copy_tile_to_dst_init_short(cb_bias);
    for (uint32_t i = 0; i < SBW; ++i) copy_tile(cb_bias, first + i, i);
    mm_block_init_short(cb_a, cb_w, false, SBW, 1, Kt);
    reconfig_data_format(cb_w, cb_a);
}

struct LnCbs {
    static constexpr uint32_t Kt = ::Kt, x = cb_x, gamma = cb_gamma, beta = cb_beta, scaler = cb_scaler,
        eps = cb_eps, mean = cb_mean, var = cb_var, rstd = cb_rstd, xc = cb_xc, tmp = cb_tmp;
};

// One 32-row tile of x (Kt tiles at the front of cb_x, popped) -> Kt bf16 tiles appended to cb_ln.
ALWI void layer_norm_row() {
    if constexpr (MOVED) {
        pack_reconfig_data_format(cb_x);
        cb_reserve_back(cb_x, Kt);
        if constexpr (CP > 0) {
            reconfig_data_format_srca(cb_prev);
            copy_tile_to_dst_init_short(cb_prev);
            cb_wait_front(cb_prev, CP);
            for (uint32_t t = 0; t < CP; ++t) {
                tile_regs_acquire();
                copy_tile(cb_prev, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_x);
                tile_regs_release();
            }
            cb_pop_front(cb_prev, CP);
        }
        reconfig_data_format_srca(cb_in);
        transpose_wh_init_short(cb_in);
        for (uint32_t t = CP; t < Kt; ++t) {
            cb_wait_front(cb_in, 1);
            tile_regs_acquire();
            transpose_wh_tile(cb_in, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_x);
            tile_regs_release();
            cb_pop_front(cb_in, 1);
        }
        cb_push_back(cb_x, Kt);
    }
    cb_reserve_back(cb_ln, Kt);
    if constexpr (ABL & 1) {
        cb_wait_front(cb_x, Kt);
        cb_pop_front(cb_x, Kt);
    } else {
        ln_prologue::layer_norm_row<LnCbs>(cb_ln, [](uint32_t) { pack_tile(0, cb_ln); });
    }
    cb_push_back(cb_ln, Kt);
}

// FOLD: rstd of row tile mi of the Mb * Kt x tiles at the front of cb_x (not popped), appended to cb_rstd.
ALWI void fold_rstd_row(uint32_t mi) {
    const uint32_t x0 = mi * Kt;
    reconfig_data_format(cb_x, cb_scaler);                      // mean
    pack_reconfig_data_format(cb_mean);
    cb_reserve_back(cb_mean, 1);
    tile_regs_acquire();
    reduce_init<REDUCE_OP, REDUCE_DIM, true>(cb_x, cb_scaler, cb_mean);
    for (uint32_t i = 0; i < Kt; ++i) reduce_tile<REDUCE_OP, REDUCE_DIM, true>(cb_x, cb_scaler, x0 + i, 0, 0);
    reduce_uninit<true>();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_mean);
    tile_regs_release();
    cb_push_back(cb_mean, 1);
    reconfig_data_format(cb_x, cb_x);                           // x^2
    pack_reconfig_data_format(cb_tmp);
    mul_tiles_init(cb_x, cb_x);
    cb_reserve_back(cb_tmp, Kt);
    for (uint32_t i0 = 0; i0 < Kt; i0 += 4) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < 4 && i0 + j < Kt; ++j) mul_tiles(cb_x, cb_x, x0 + i0 + j, x0 + i0 + j, j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < 4 && i0 + j < Kt; ++j) pack_tile(j, cb_tmp);
        tile_regs_release();
    }
    cb_push_back(cb_tmp, Kt);
    ln_prologue::row_mean<LnCbs>(cb_tmp, cb_var, true);         // E[x^2]
    cb_wait_front(cb_mean, 1);                                  // rstd = rsqrt(E[x^2] - mean^2 + eps)
    cb_wait_front(cb_var, 1);
    reconfig_data_format(cb_mean, cb_mean);
    pack_reconfig_data_format(cb_rstd);
    cb_reserve_back(cb_rstd, 1);
    tile_regs_acquire();
    mul_tiles_init(cb_mean, cb_mean);
    mul_tiles(cb_mean, cb_mean, 0, 0, 0);
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_var);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_var, 0, 0);
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_eps);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_eps, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rstd);
    tile_regs_release();
    cb_push_back(cb_rstd, 1);
    cb_pop_front(cb_mean, 1);
    cb_pop_front(cb_var, 1);
}

void kernel_main() {
    const uint32_t nblocks = get_arg_val<uint32_t>(0);
    constexpr uint32_t blk = Mb * Nt;

    mm_init(cb_ln, cb_w, cb_p);
    cb_wait_front(cb_w, (1 + GATE) * Kt * Nt);
    cb_wait_front(cb_gamma, Kt);
    cb_wait_front(cb_beta, Kt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    if constexpr (NBIAS > 0) cb_wait_front(cb_bias, NBIAS * Nt);

    for (uint32_t b = 0; b < nblocks; ++b) {
        // 1. LayerNorm prologue, row tile by row tile (FOLD: rstd only).
        constexpr uint32_t cb_a = FOLD ? cb_x : cb_ln;
        if constexpr (FOLD) {
            cb_wait_front(cb_x, Mb * Kt);
            for (uint32_t mi = 0; mi < Mb; ++mi) fold_rstd_row(mi);
        } else {
            for (uint32_t mi = 0; mi < Mb; ++mi) {
                layer_norm_row();
            }
        }

        // 2. p = x_norm @ Wp (FOLD: x @ Wc), one K block accumulated in DST in k order, packed to bf16.
        mm_fid::block_init_short<MM_FID>(cb_a, cb_w, SBW, 1, Kt);
        reconfig_data_format(cb_w, cb_a);
        pack_reconfig_data_format(cb_p);
        cb_wait_front(cb_a, Mb * Kt);
        cb_reserve_back(cb_p, blk);
        for (uint32_t mi = 0; mi < ((ABL & 2) ? 0 : Mb); ++mi) {
            for (uint32_t n0 = 0; n0 < Nt; n0 += SBW) {
                tile_regs_acquire();
                if constexpr (EXTRA & 1) bias_to_dst(n0, cb_ln);
                uint32_t i0 = mi * Kt;
                uint32_t i1 = n0;
                for (uint32_t k = 0; k < Kt; ++k) {
                    mm_fid::block<MM_FID>(cb_a, cb_w, i0, i1, 0, SBW, 1, Kt);
                    i0++;
                    i1 += Nt;
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < SBW; ++i) pack_tile<true>(i, cb_p, mi * Nt + n0 + i);
                tile_regs_release();
            }
        }
        cb_push_back(cb_p, blk);
        cb_pop_front(cb_a, Mb * Kt);

        // 2b (GATE). g = sigmoid(xb @ Wg), the sigmoid in DST before the pack. Its init follows the
        // GEMM init: the prologue's rsqrt init reprograms the SFPU every block.
        if constexpr (GATE) {
            mm_block_init_short(cb_xb, cb_w, false, SBW, 1, Kt);
            reconfig_data_format(cb_w, cb_xb);
            pack_reconfig_data_format(cb_g);
            sigmoid_tile_init();
            cb_wait_front(cb_xb, Mb * Kt);
            cb_reserve_back(cb_g, blk);
            for (uint32_t mi = 0; mi < ((ABL & 4) ? 0 : Mb); ++mi) {
                for (uint32_t n0 = 0; n0 < Nt; n0 += SBW) {
                    tile_regs_acquire();
                    if constexpr (EXTRA & 2) {
                        bias_to_dst((EXTRA & 1) * Nt + n0, cb_xb);
                        sigmoid_tile_init();
                    }
                    uint32_t i0 = mi * Kt;
                    uint32_t i1 = Kt * Nt + n0;
                    for (uint32_t k = 0; k < Kt; ++k) {
                        matmul_block(cb_xb, cb_w, i0, i1, 0, false, SBW, 1, Kt);
                        i0++;
                        i1 += Nt;
                    }
                    for (uint32_t i = 0; i < SBW; ++i) {
                        if constexpr (SIGPOLY) sigmoid_poly_tile<SIGPOLY>(i);
                        else sigmoid_bf16_tile(i);
                    }
                    if constexpr (RNE & 2) {
                        for (uint32_t i = 0; i < SBW; ++i) round_bf16_rne_tile(i);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < SBW; ++i) pack_tile<true>(i, cb_g, mi * Nt + n0 + i);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_g, blk);
            cb_pop_front(cb_xb, Mb * Kt);
        }

        // 3 + 4. out = z + p * sigmoid(g). The sigmoid is re-initialised every block: the
        // prologue's rsqrt init reprograms the SFPU.
        cb_wait_front(cb_p, blk);
        cb_wait_front(cb_g, blk);
        cb_wait_front(cb_z, blk);
        if constexpr (EXTRA & 4) cb_wait_front(cb_mask, Nt);
        cb_reserve_back(cb_out, blk);
#ifdef T2_RES32
        cb_reserve_back(cb_copy, blk);
#endif
        sigmoid_tile_init();
        pack_reconfig_data_format(cb_out);
        if constexpr (FOLD) cb_wait_front(cb_rstd, Mb);
        for (uint32_t t0 = 0; t0 < ((ABL & 8) ? 0 : blk); t0 += G) {
            tile_regs_acquire();
#ifdef T2_RES32
            sigmoid_tile_init();    // the SFPU add below reprograms it
#endif
            if constexpr (FOLD) {
                // DST = p * rstd + bias, then * g below in the same DST (g is already a sigmoid under GATE).
                reconfig_data_format(cb_p, cb_rstd);
                mul_bcast_cols_init_short(cb_p, cb_rstd);
                for (uint32_t i = 0; i < 4; ++i) mul_tiles_bcast_cols(cb_p, cb_rstd, t0 + i, (t0 + i) / Nt, i);
                reconfig_data_format(cb_beta, cb_beta);
                binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_beta);
                for (uint32_t i = 0; i < 4; ++i)
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        cb_beta, (t0 + i) % Nt, i);
                reconfig_data_format(cb_g, cb_g);
                binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_g);
                for (uint32_t i = 0; i < 4; ++i)
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        cb_g, t0 + i, i);
            } else {
            reconfig_data_format_srca(cb_g);
            copy_tile_to_dst_init_short(cb_g);
            for (uint32_t i = 0; i < G; ++i) copy_tile(cb_g, t0 + i, i);
            if constexpr (!GATE && !(ABL & 16)) {
                for (uint32_t i = 0; i < G; ++i) {
                    if constexpr (SIGPOLY) sigmoid_poly_tile<SIGPOLY>(i);
                    else sigmoid_bf16_tile(i);
                }
            }
            reconfig_data_format(cb_p, cb_p);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_p);
            for (uint32_t i = 0; i < G; ++i)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_p, t0 + i, i);
            }
            if constexpr (FOLD) reconfig_data_format(cb_z, cb_z);
            if constexpr (RNE & 4) {
                for (uint32_t i = 0; i < G; ++i) round_bf16_rne_tile(i);
            }
            if constexpr (EXTRA & 4) {
                binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_mask);
                for (uint32_t i = 0; i < G; ++i)
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        cb_mask, (t0 + i) % Nt, i);
                if constexpr (RNE & 4) {
                    for (uint32_t i = 0; i < G; ++i) round_bf16_rne_tile(i);
                }
            }
#ifdef T2_RES32
            reconfig_data_format_srca(cb_z);
            copy_tile_to_dst_init_short(cb_z);
            for (uint32_t i = 0; i < G; ++i) copy_tile(cb_z, t0 + i, G + i);
            add_binary_tile_init();
            for (uint32_t i = 0; i < G; ++i) add_binary_tile(i, G + i, i);
#else
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_z);
            for (uint32_t i = 0; i < G; ++i)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_z, t0 + i, i);
            if constexpr (RNE & 1) {
                for (uint32_t i = 0; i < G; ++i) round_bf16_rne_tile(i);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
#ifdef T2_RES32
            pack_reconfig_data_format(cb_out);
            for (uint32_t i = 0; i < G; ++i) pack_tile(i, cb_out);
            pack_reconfig_data_format(cb_copy);
            for (uint32_t i = 0; i < G; ++i) pack_tile(i, cb_copy);
#else
            for (uint32_t i = 0; i < 4; ++i) pack_tile(i, cb_out);
#endif
            tile_regs_release();
        }
        cb_push_back(cb_out, blk);
#ifdef T2_RES32
        cb_push_back(cb_copy, blk);
#endif
        cb_pop_front(cb_p, blk);
        cb_pop_front(cb_g, blk);
        cb_pop_front(cb_z, blk);
        if constexpr (EXTRA & 4) cb_pop_front(cb_mask, Nt);
        if constexpr (FOLD) cb_pop_front(cb_rstd, Mb);
    }
}
