// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ln_split compute (tt_bio/ln_split.py): the LayerNorm of this core's KC column tiles of one row tile, with row
// statistics over all KT tiles of the row from the G cores that share it. Two-pass as ttnn.layer_norm: the partial
// row sum (x 1/K) goes to the writer, the G partials come back and their sum is the mean; then the same for the
// biased variance of x - mean. Sums of partials are SFPU adds in fp32 DEST. Then (x - mean) * rstd [* gamma] [+ beta]
// in one DEST pass (normalize_out), rounded to nearest even before the bf16 pack. x - mean and its square come out
// of one DEST acquire (the square on the SFPU in fp32), not a second pass that re-reads x - mean through srcA.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "../trimul_gin_moved/sigmoid_poly.hpp"
#include "../common/ln_prologue.hpp"

constexpr uint32_t KC = get_compile_time_arg_val(0);
constexpr uint32_t G = get_compile_time_arg_val(1);
constexpr uint32_t HAS_W = get_compile_time_arg_val(2);
constexpr uint32_t HAS_B = get_compile_time_arg_val(3);

constexpr uint32_t cb_out = tt::CBIndex::c_2;
constexpr uint32_t cb_x = tt::CBIndex::c_7;
constexpr uint32_t cb_gamma = tt::CBIndex::c_8;
constexpr uint32_t cb_beta = tt::CBIndex::c_9;
constexpr uint32_t cb_scaler = tt::CBIndex::c_10;
constexpr uint32_t cb_eps = tt::CBIndex::c_11;
constexpr uint32_t cb_p1 = tt::CBIndex::c_12;     // partial row sum / K, to the writer
constexpr uint32_t cb_parts1 = tt::CBIndex::c_13; // the G partials
constexpr uint32_t cb_mean = tt::CBIndex::c_14;
constexpr uint32_t cb_p2 = tt::CBIndex::c_15;
constexpr uint32_t cb_xc = tt::CBIndex::c_16;
constexpr uint32_t cb_tmp = tt::CBIndex::c_17;
constexpr uint32_t cb_parts2 = tt::CBIndex::c_18;
constexpr uint32_t cb_var = tt::CBIndex::c_19;
constexpr uint32_t cb_rstd = tt::CBIndex::c_20;

struct LnCbs {
    static constexpr uint32_t Kt = KC, x = cb_x, gamma = cb_gamma, beta = cb_beta, scaler = cb_scaler, eps = cb_eps,
        mean = cb_mean, var = cb_var, rstd = cb_rstd, xc = cb_xc, tmp = cb_tmp;
};

// out = sum of the G tiles of `in` (popped), in fp32 DEST.
ALWI void sum_parts(uint32_t in, uint32_t out) {
    cb_wait_front(in, G);
    cb_reserve_back(out, 1);
    reconfig_data_format_srca(in);
    pack_reconfig_data_format(out);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(in);
    copy_tile(in, 0, 0);
    for (uint32_t k = 1; k < G; ++k) {
        copy_tile_to_dst_init_short(in);
        copy_tile(in, k, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    cb_pop_front(in, G);
}

// xc = x - mean and tmp = xc^2, KC tiles each, two tiles a DEST acquire (four fp32 slots of a half-sync dest). The
// subtraction runs twice per tile rather than an SFPU dest copy, which not every tt-metal here has.
ALWI void center_and_square() {
    reconfig_data_format(cb_x, cb_mean);
    pack_reconfig_data_format(cb_xc);
    cb_reserve_back(cb_xc, KC);
    cb_reserve_back(cb_tmp, KC);
    for (uint32_t i0 = 0; i0 < KC; i0 += 2) {
        const uint32_t n = KC - i0 < 2 ? KC - i0 : 2;
        tile_regs_acquire();
        sub_bcast_cols_init_short(cb_x, cb_mean);
        for (uint32_t j = 0; j < n; ++j) {
            sub_tiles_bcast_cols(cb_x, cb_mean, i0 + j, 0, j);
            sub_tiles_bcast_cols(cb_x, cb_mean, i0 + j, 0, 2 + j);
        }
        square_tile_init();
        for (uint32_t j = 0; j < n; ++j) square_tile(2 + j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) pack_tile(j, cb_xc);
        for (uint32_t j = 0; j < n; ++j) pack_tile(2 + j, cb_tmp);
        tile_regs_release();
    }
    cb_push_back(cb_xc, KC);
    cb_push_back(cb_tmp, KC);
}

// out = xc * rstd [* gamma] [+ beta], two tiles a DEST acquire, one rounded pack. Every operand reaches DEST whole:
// the fp32 xc and rstd by unpack-to-dest (unary_bcast does that for fp32 CBs; srcA / srcB would cut them to tf32),
// rstd broadcast along the row and gamma / beta down the column by the unpacker, the products and sum on the SFPU in
// fp32. Was three FPU stages, each a pack and a tf32 re-read of the last stage's fp32 output.
ALWI void normalize_out() {
    if constexpr (HAS_W) cb_wait_front(cb_gamma, KC);   // read after x: the statistics do not wait for it
    if constexpr (HAS_B) cb_wait_front(cb_beta, KC);
    for (uint32_t i0 = 0; i0 < KC; i0 += 2) {
        const uint32_t n = KC - i0 < 2 ? KC - i0 : 2;
        cb_reserve_back(cb_out, n);
        tile_regs_acquire();
        unary_bcast_init<BroadcastType::COL>(cb_rstd, cb_out);
        unary_bcast<BroadcastType::COL>(cb_rstd, 0, 2);
        unary_bcast_init<BroadcastType::NONE>(cb_xc, cb_out);
        for (uint32_t j = 0; j < n; ++j) unary_bcast<BroadcastType::NONE>(cb_xc, i0 + j, j);
        mul_binary_tile_init();
        for (uint32_t j = 0; j < n; ++j) mul_binary_tile(j, 2, j);
        if constexpr (HAS_W) {
            for (uint32_t j = 0; j < n; ++j) {
                unary_bcast_init<BroadcastType::ROW>(cb_gamma, cb_out);
                unary_bcast<BroadcastType::ROW>(cb_gamma, i0 + j, 3);
                mul_binary_tile_init();
                mul_binary_tile(j, 3, j);
            }
        }
        if constexpr (HAS_B) {
            for (uint32_t j = 0; j < n; ++j) {
                unary_bcast_init<BroadcastType::ROW>(cb_beta, cb_out);
                unary_bcast<BroadcastType::ROW>(cb_beta, i0 + j, 3);
                add_binary_tile_init();
                add_binary_tile(j, 3, j);
            }
        }
        for (uint32_t j = 0; j < n; ++j) round_bf16_rne_tile(j);   // the packer truncates; ttnn.layer_norm's output rounds
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) pack_tile(j, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, n);
    }
}

void kernel_main() {
    binary_op_init_common(cb_x, cb_scaler, cb_p1);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    ln_prologue::row_mean<LnCbs>(cb_x, cb_p1, false);         // partial sum / K
    sum_parts(cb_parts1, cb_mean);                        // mean
    cb_wait_front(cb_mean, 1);
    center_and_square();
    cb_pop_front(cb_x, KC);
    cb_pop_front(cb_mean, 1);
    cb_wait_front(cb_xc, KC);
    ln_prologue::row_mean<LnCbs>(cb_tmp, cb_p2, true);        // partial sum (x - mean)^2 / K
    sum_parts(cb_parts2, cb_var);                         // biased variance
    ln_prologue::rstd<LnCbs>();
    normalize_out();
    cb_pop_front(cb_xc, KC);
    cb_pop_front(cb_rstd, 1);
}
}
