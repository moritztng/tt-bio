// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul_fused compute (bc4-bet-fusedbuild), phase 1 of the fused AF2 trimul forward. Per half-cube unit (16 rows r
// at column tile s):
//   1. LayerNorm in of the 16 row tiles (ln_prologue, two-pass fp32 statistics) into c_2, kept for the unit.
//   2. Per channel tile q and half (a, b): for each row li, P = bias[q] + W_p^T[q] @ x_li^T and G likewise in DST
//      (in1 transposed in the unpacker, so P is [32 c x 32 s]); P *= sigmoid(G) in DST; a also times the mask row.
//      The tile is packed untilized at column 16 h + li of a 32-tile row in c_5, so c_5's 2 KB page c is channel
//      c's [r x s] block in row-major order (dit_pair_bias's rotation).
//   3. Each page tilized alone into c_6: one [r x s] tile per channel, rows 16 h .. 16 h + 15 real. The writer
//      sends those 1 KB (faces 2h, 2h + 1).
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "../trimul_gin_moved/sigmoid_poly.hpp"
#include "../common/ln_prologue.hpp"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t MASK = get_compile_time_arg_val(1);
// SIG 1 / 2: sigmoid_poly.hpp's degree-8 / degree-6 polynomial (1 is today's default); 3: metal's sigmoid in its
// approximate (fast) SFPU mode, the mode bc4-bet-sigmoid grades.
constexpr uint32_t SIG = get_compile_time_arg_val(2);
// ABL (bench only): bit 0 no LayerNorm (z copied), bit 1 no matmul, bit 2 no gate / mask, bit 3 no tilize.
constexpr uint32_t ABL = get_compile_time_arg_val(3);

constexpr uint32_t cb_w = 0, cb_z = 1, cb_xn = 2, cb_bias = 3, cb_mask = 4, cb_rm = 5, cb_t = 6;
constexpr uint32_t cb_gamma = 8, cb_beta = 9, cb_scaler = 10, cb_eps = 11, cb_mean = 12, cb_var = 13, cb_rstd = 14;
constexpr uint32_t cb_xc = 16, cb_tmp = 17;

struct LnCbs {
    static constexpr uint32_t Kt = ::Kt, x = cb_z, gamma = cb_gamma, beta = cb_beta, scaler = cb_scaler,
        eps = cb_eps, mean = cb_mean, var = cb_var, rstd = cb_rstd, xc = cb_xc, tmp = cb_tmp;
};

void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    mm_init(cb_w, cb_xn, cb_rm, 1);
    cb_wait_front(cb_w, 16 * Kt);
    cb_wait_front(cb_bias, 16);
    cb_wait_front(cb_gamma, Kt);
    cb_wait_front(cb_beta, Kt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t h = u % 2;
        // 1. LayerNorm in.
        cb_reserve_back(cb_xn, 16 * Kt);
        for (uint32_t li = 0; li < 16; ++li) {
            if constexpr (ABL & 1) {
                cb_wait_front(cb_z, Kt);
                reconfig_data_format_srca(cb_z);
                pack_reconfig_data_format(cb_xn);
                copy_tile_to_dst_init_short(cb_z);
                for (uint32_t i = 0; i < Kt; ++i) {
                    tile_regs_acquire();
                    copy_tile(cb_z, i, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_xn);
                    tile_regs_release();
                }
                cb_pop_front(cb_z, Kt);
            } else {
                ln_prologue::layer_norm_row<LnCbs>(cb_xn, [](uint32_t) { pack_tile(0, cb_xn); });
            }
        }
        cb_push_back(cb_xn, 16 * Kt);
        cb_wait_front(cb_xn, 16 * Kt);
        if constexpr (MASK) cb_wait_front(cb_mask, 16);

        // 2-3. Projections, gates, rotation.
        for (uint32_t q = 0; q < 4; ++q) {
            for (uint32_t ab = 0; ab < 2; ++ab) {
                const uint32_t wp = (2 * ab) * 4 + q, wg = (2 * ab + 1) * 4 + q;
                reconfig_data_format(cb_w, cb_xn);
                pack_reconfig_data_format(cb_rm);
                cb_reserve_back(cb_rm, 32);
                pack_untilize_dest_init<1, 32>(cb_rm);
                for (uint32_t li = 0; li < 16; ++li) {
                    tile_regs_acquire();
                    copy_tile_to_dst_init_short(cb_bias);
                    copy_tile(cb_bias, wp, 0);
                    copy_tile(cb_bias, wg, 1);
                    if constexpr (!(ABL & 2)) {
                        mm_init_short(cb_w, cb_xn, 1);
                        for (uint32_t kc = 0; kc < Kt; ++kc) {
                            matmul_tiles(cb_w, cb_xn, wp * Kt + kc, li * Kt + kc, 0);
                            matmul_tiles(cb_w, cb_xn, wg * Kt + kc, li * Kt + kc, 1);
                        }
                    }
                    if constexpr (!(ABL & 4)) {
                        if constexpr (SIG == 3) {
                            sigmoid_tile_init<true>();
                            sigmoid_tile<VectorMode::RC, true>(1);
                            mul_binary_tile_init();
                            mul_binary_tile(0, 1, 0);
                        } else {
                            gate_poly_tile<SIG, 1>(0);
                        }
                        if (MASK && ab == 0) {
                            copy_tile_to_dst_init_short(cb_mask);
                            copy_tile(cb_mask, li, 1);
                            mul_binary_tile_init();
                            mul_binary_tile(0, 1, 0);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_untilize_dest<1, 32>(cb_rm, 1, 16 * h + li);
                    tile_regs_release();
                }
                pack_untilize_uninit(cb_rm);
                cb_push_back(cb_rm, 32);
                if constexpr (ABL & 8) {
                    cb_wait_front(cb_rm, 32);
                    cb_pop_front(cb_rm, 32);
                    for (uint32_t c = 0; c < 32; ++c) {
                        cb_reserve_back(cb_t, 1);
                        cb_push_back(cb_t, 1);
                    }
                    continue;
                }
                reconfig_data_format_srca(cb_rm);
                pack_reconfig_data_format(cb_t);
                tilize_init(cb_rm, 1, cb_t);
                for (uint32_t c = 0; c < 32; ++c) {
                    cb_wait_front(cb_rm, 1);
                    cb_reserve_back(cb_t, 1);
                    tilize_block(cb_rm, 1, cb_t);
                    cb_push_back(cb_t, 1);
                    cb_pop_front(cb_rm, 1);
                }
                tilize_uninit(cb_rm, cb_t);
                mm_init_short(cb_w, cb_xn, 1);
            }
        }
        cb_pop_front(cb_xn, 16 * Kt);
        if constexpr (MASK) cb_pop_front(cb_mask, 16);
    }
}
