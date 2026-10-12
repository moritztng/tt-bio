// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Transition T3: the whole SwiGLU transition of one row tile of x in L1, weights resident.
//   1. xn = layer_norm(x) (kernels/common/ln_prologue.hpp), Kt bf16 tiles.
//   2. per group of G hidden tiles: fc1 block -> silu in DST -> gate CB; fc2 block -> * gate (DST reused as
//      srcB) -> h slots [g*G, g*G + G). Neither fc1 nor fc2 leaves L1 as a tensor.
//   3. y = h @ W3, all Ht hidden tiles accumulated in one fp32 DST half, the Kt output tiles in groups of 4
//      (two groups at c 256, Protenix-v2).
//   4. RES: y packed bf16, then y + x (x from the residual CB) packed bf16 to the writer.
// fp32 DEST (the statistics and the K accumulation), the caller's fidelity; under math_approx_mode silu is
// the silu_f32 overlay (tenstorrent.silu_ckc), as the incumbent fc1 runs it.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "../common/ln_prologue.hpp"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Ht = get_compile_time_arg_val(1);
constexpr uint32_t RES = get_compile_time_arg_val(2);
constexpr uint32_t F3S = get_compile_time_arg_val(3);       // fc3 K chunks: partials via fp32 cb_part between them
static_assert(Ht % F3S == 0, "fc3 chunks divide the hidden tiles");
constexpr uint32_t G = 4;                                   // fp32 half-sync DST holds 4 tiles
constexpr uint32_t NG = Kt < G ? Kt : G;                    // fc3 output tiles per DST half
static_assert(Ht % G == 0 && Kt % NG == 0, "hidden in groups of 4 tiles, fc3 output in groups of NG");
static_assert(F3S == 1 || Kt <= 4, "fc3 K chunks only where one DST half holds the output row tile");

constexpr uint32_t w12_cb = tt::CBIndex::c_0;    // [W1 | W2], Kt x 2Ht, resident
constexpr uint32_t w3_cb = tt::CBIndex::c_1;     // W3, Ht x Kt, resident
constexpr uint32_t xn_cb = tt::CBIndex::c_2;     // layer_norm(x), Kt
constexpr uint32_t gate_cb = tt::CBIndex::c_3;   // silu(fc1) of one group, G
constexpr uint32_t h_cb = tt::CBIndex::c_4;      // silu(fc1) * fc2, Ht
constexpr uint32_t res_cb = tt::CBIndex::c_5;    // x again, for the residual
constexpr uint32_t out_cb = tt::CBIndex::c_6;
constexpr uint32_t cb_x = tt::CBIndex::c_7;
constexpr uint32_t cb_gamma = tt::CBIndex::c_8;
constexpr uint32_t cb_beta = tt::CBIndex::c_9;
constexpr uint32_t cb_scaler = tt::CBIndex::c_10;
constexpr uint32_t cb_eps = tt::CBIndex::c_11;
constexpr uint32_t cb_mean = tt::CBIndex::c_12;
constexpr uint32_t cb_var = tt::CBIndex::c_13;
constexpr uint32_t cb_rstd = tt::CBIndex::c_14;
constexpr uint32_t cb_xc = tt::CBIndex::c_16;
constexpr uint32_t cb_tmp = tt::CBIndex::c_17;
constexpr uint32_t cb_part = tt::CBIndex::c_18;  // fc3 fp32 partial between K chunks (F3S > 1)
constexpr uint32_t cb_y = tt::CBIndex::c_15;     // fc3 output, bf16, before the residual add

struct LnCbs {
    static constexpr uint32_t Kt = ::Kt, x = cb_x, gamma = cb_gamma, beta = cb_beta, scaler = cb_scaler,
        eps = cb_eps, mean = cb_mean, var = cb_var, rstd = cb_rstd, xc = cb_xc, tmp = cb_tmp;
};

void kernel_main() {
    const uint32_t nunits = get_arg_val<uint32_t>(0);

    mm_init(xn_cb, w12_cb, gate_cb);
    cb_wait_front(w12_cb, 2 * Ht * Kt);
    cb_wait_front(w3_cb, Ht * Kt);
    cb_wait_front(cb_gamma, Kt);
    cb_wait_front(cb_beta, Kt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    for (uint32_t u = 0; u < nunits; ++u) {
        // 1. layer_norm
        cb_reserve_back(xn_cb, Kt);
        ln_prologue::layer_norm_row<LnCbs>(xn_cb, [](uint32_t i) { pack_tile<true>(0, xn_cb, i); });
        cb_push_back(xn_cb, Kt);
        cb_wait_front(xn_cb, Kt);

        // 2. h = silu(xn @ W1) * (xn @ W2), G hidden tiles at a time. silu is re-initialised every unit:
        // the prologue's rsqrt init reprograms the SFPU.
        silu_tile_init();
        cb_reserve_back(h_cb, Ht);
        for (uint32_t g = 0; g < Ht; g += G) {
            mm_block_init_short(xn_cb, w12_cb, false, G, 1, Kt);
            reconfig_data_format(w12_cb, xn_cb);
            pack_reconfig_data_format(gate_cb);
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; ++k) matmul_block(xn_cb, w12_cb, k, k * 2 * Ht + g, 0, false, G, 1, Kt);
            for (uint32_t i = 0; i < G; ++i) silu_tile(i);
            tile_regs_commit();
            cb_reserve_back(gate_cb, G);
            tile_regs_wait();
            for (uint32_t i = 0; i < G; ++i) pack_tile(i, gate_cb);
            tile_regs_release();
            cb_push_back(gate_cb, G);

            mm_block_init_short(xn_cb, w12_cb, false, G, 1, Kt);
            reconfig_data_format(w12_cb, xn_cb);
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; ++k)
                matmul_block(xn_cb, w12_cb, k, k * 2 * Ht + Ht + g, 0, false, G, 1, Kt);
            cb_wait_front(gate_cb, G);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(gate_cb);
            for (uint32_t i = 0; i < G; ++i)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    gate_cb, i, i);
            tile_regs_commit();
            pack_reconfig_data_format(h_cb);
            tile_regs_wait();
            for (uint32_t i = 0; i < G; ++i) pack_tile<true>(i, h_cb, g + i);
            tile_regs_release();
            cb_pop_front(gate_cb, G);
        }
        cb_push_back(h_cb, Ht);
        cb_pop_front(xn_cb, Kt);
        cb_wait_front(h_cb, Ht);

        // 3. y = h @ W3, one K block of Ht tiles, then 4. + x.
        constexpr uint32_t Hc = Ht / F3S;
        for (uint32_t c = 0; c + 1 < F3S; ++c) {
            tile_regs_acquire();
            if (c > 0) {
                cb_wait_front(cb_part, Kt);
                copy_tile_to_dst_init_short(cb_part);
                reconfig_data_format_srca(cb_part);
                for (uint32_t i = 0; i < Kt; ++i) copy_tile(cb_part, i, i);
                cb_pop_front(cb_part, Kt);
            }
            mm_block_init_short(h_cb, w3_cb, false, Kt, 1, Ht);
            reconfig_data_format(w3_cb, h_cb);
            for (uint32_t j = c * Hc; j < (c + 1) * Hc; ++j) matmul_block(h_cb, w3_cb, j, j * Kt, 0, false, Kt, 1, Ht);
            tile_regs_commit();
            pack_reconfig_data_format(cb_part);
            cb_reserve_back(cb_part, Kt);
            tile_regs_wait();
            for (uint32_t i = 0; i < Kt; ++i) pack_tile(i, cb_part);
            tile_regs_release();
            cb_push_back(cb_part, Kt);
        }
        const uint32_t y_cb = RES ? cb_y : out_cb;
        cb_reserve_back(y_cb, Kt);
        for (uint32_t n0 = 0; n0 < Kt; n0 += NG) {
            tile_regs_acquire();
            if constexpr (F3S > 1) {
                cb_wait_front(cb_part, Kt);
                copy_tile_to_dst_init_short(cb_part);
                reconfig_data_format_srca(cb_part);
                for (uint32_t i = 0; i < Kt; ++i) copy_tile(cb_part, i, i);
                cb_pop_front(cb_part, Kt);
            }
            mm_block_init_short(h_cb, w3_cb, false, NG, 1, Ht);
            reconfig_data_format(w3_cb, h_cb);
            pack_reconfig_data_format(y_cb);
            for (uint32_t j = (F3S - 1) * Hc; j < Ht; ++j)
                matmul_block(h_cb, w3_cb, j, j * Kt + n0, 0, false, NG, 1, Ht);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < NG; ++i) pack_tile<true>(i, y_cb, n0 + i);
            tile_regs_release();
        }
        cb_push_back(y_cb, Kt);
        // 4. + x: y is packed bf16 and added to x as two bf16 operands, the incumbent's fc3 -> add_ rounding.
        // (Adding x into the fp32 DST by DEST_TO_SRCB reuse put 9 of 33.5M elements exactly 1 or 2 low at 512,
        // the same elements on every chip and run.)
        if constexpr (RES) {
            cb_wait_front(cb_y, Kt);
            cb_wait_front(res_cb, Kt);
            reconfig_data_format(cb_y, res_cb);
            pack_reconfig_data_format(out_cb);
            add_tiles_init(cb_y, res_cb);
            cb_reserve_back(out_cb, Kt);
            for (uint32_t n0 = 0; n0 < Kt; n0 += NG) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < NG; ++i) add_tiles(cb_y, res_cb, n0 + i, n0 + i, i);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < NG; ++i) pack_tile(i, out_cb);
                tile_regs_release();
            }
            cb_pop_front(cb_y, Kt);
            cb_push_back(out_cb, Kt);
            cb_pop_front(res_cb, Kt);
        }
        cb_pop_front(h_cb, Ht);
    }
}
