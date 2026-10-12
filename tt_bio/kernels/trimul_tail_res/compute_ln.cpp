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
#include "../trimul_gin_moved/sigmoid_poly.hpp"
#include "../common/ln_prologue.hpp"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Nt = get_compile_time_arg_val(1);
constexpr uint32_t Mb = get_compile_time_arg_val(2);
constexpr uint32_t SBW = get_compile_time_arg_val(3);
constexpr uint32_t SIGPOLY = get_compile_time_arg_val(4);   // 1: sigmoid_poly.hpp's polynomial
// Round DST to bf16 nearest-even: bit 0 before the output pack, bit 2 the product before the
// residual add (as the res compute's bits 0 / 2).
constexpr uint32_t RNE = get_compile_time_arg_val(5);

constexpr uint32_t cb_x = tt::CBIndex::c_0;      // x_out rows, bf16, Mb * Kt double buffered
constexpr uint32_t cb_w = tt::CBIndex::c_1;      // Wp, Kt * Nt, resident
constexpr uint32_t cb_out = tt::CBIndex::c_2;    // z + update, to the writer
constexpr uint32_t cb_ln = tt::CBIndex::c_3;     // layer_norm(x) rows, bf16, Mb * Kt
constexpr uint32_t cb_p = tt::CBIndex::c_4;      // x_norm @ Wp, bf16, Mb * Nt
constexpr uint32_t cb_g = tt::CBIndex::c_5;      // g_out rows, bf16, from DRAM
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

ALWI void sigmoid_bf16_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid<false, false>(idst, (int)VectorMode::RC)));
}

struct LnCbs {
    static constexpr uint32_t Kt = ::Kt, x = cb_x, gamma = cb_gamma, beta = cb_beta, scaler = cb_scaler,
        eps = cb_eps, mean = cb_mean, var = cb_var, rstd = cb_rstd, xc = cb_xc, tmp = cb_tmp;
};

// One 32-row tile of x (Kt tiles at the front of cb_x, popped) -> Kt bf16 tiles appended to cb_ln.
ALWI void layer_norm_row() {
    cb_reserve_back(cb_ln, Kt);
    ln_prologue::layer_norm_row<LnCbs>(cb_ln, [](uint32_t) { pack_tile(0, cb_ln); });
    cb_push_back(cb_ln, Kt);
}

void kernel_main() {
    const uint32_t nblocks = get_arg_val<uint32_t>(0);
    constexpr uint32_t blk = Mb * Nt;

    mm_init(cb_ln, cb_w, cb_p);
    cb_wait_front(cb_w, Kt * Nt);
    cb_wait_front(cb_gamma, Kt);
    cb_wait_front(cb_beta, Kt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    for (uint32_t b = 0; b < nblocks; ++b) {
        // 1. LayerNorm prologue, row tile by row tile.
        for (uint32_t mi = 0; mi < Mb; ++mi) layer_norm_row();

        // 2. p = x_norm @ Wp, one K block accumulated in DST in k order, packed to bf16.
        mm_block_init_short(cb_ln, cb_w, false, SBW, 1, Kt);
        reconfig_data_format(cb_w, cb_ln);
        pack_reconfig_data_format(cb_p);
        cb_wait_front(cb_ln, Mb * Kt);
        cb_reserve_back(cb_p, blk);
        for (uint32_t mi = 0; mi < Mb; ++mi) {
            for (uint32_t n0 = 0; n0 < Nt; n0 += SBW) {
                tile_regs_acquire();
                uint32_t i0 = mi * Kt;
                uint32_t i1 = n0;
                for (uint32_t k = 0; k < Kt; ++k) {
                    matmul_block(cb_ln, cb_w, i0, i1, 0, false, SBW, 1, Kt);
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
        cb_pop_front(cb_ln, Mb * Kt);

        // 3 + 4. out = z + p * sigmoid(g). The sigmoid is re-initialised every block: the
        // prologue's rsqrt init reprograms the SFPU.
        cb_wait_front(cb_p, blk);
        cb_wait_front(cb_g, blk);
        cb_wait_front(cb_z, blk);
        cb_reserve_back(cb_out, blk);
        sigmoid_tile_init();
        pack_reconfig_data_format(cb_out);
        for (uint32_t t0 = 0; t0 < blk; t0 += 4) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_g);
            copy_tile_to_dst_init_short(cb_g);
            for (uint32_t i = 0; i < 4; ++i) copy_tile(cb_g, t0 + i, i);
            for (uint32_t i = 0; i < 4; ++i) {
                if constexpr (SIGPOLY) sigmoid_poly_tile(i);
                else sigmoid_bf16_tile(i);
            }
            reconfig_data_format(cb_p, cb_p);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_p);
            for (uint32_t i = 0; i < 4; ++i)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_p, t0 + i, i);
            if constexpr (RNE & 4) {
                for (uint32_t i = 0; i < 4; ++i) round_bf16_rne_tile(i);
            }
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_z);
            for (uint32_t i = 0; i < 4; ++i)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_z, t0 + i, i);
            if constexpr (RNE & 1) {
                for (uint32_t i = 0; i < 4; ++i) round_bf16_rne_tile(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < 4; ++i) pack_tile(i, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, blk);
        cb_pop_front(cb_p, blk);
        cb_pop_front(cb_g, blk);
        cb_pop_front(cb_z, blk);
    }
}
