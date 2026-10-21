// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ln_split compute (tt_bio/ln_split.py): the LayerNorm of this core's KC column tiles of one row tile, with row
// statistics over all KT tiles of the row from the G cores that share it. Two-pass as ttnn.layer_norm: the partial
// row sum (x 1/K) goes to the writer, the G partials come back and their sum is the mean; then the same for the
// biased variance of x - mean. Sums of partials are SFPU adds in fp32 DEST. Then (x - mean) * rstd [* gamma] [+ beta]
// to c_2 (ln_prologue.hpp's stages), rounded to nearest even before the bf16 pack.
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

// `in` (KC tiles, popped) + beta, or a copy of it, packed to c_2.
ALWI void finish(uint32_t in) {
    cb_wait_front(in, KC);
    pack_reconfig_data_format(cb_out);
    if constexpr (HAS_B) {
        reconfig_data_format(in, cb_beta);
        add_bcast_rows_init_short(in, cb_beta);
    } else {
        reconfig_data_format_srca(in);
        copy_tile_to_dst_init_short(in);
    }
    for (uint32_t i = 0; i < KC; ++i) {   // one tile at a time, so the writer's DRAM writes overlap this stage
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        if constexpr (HAS_B) add_tiles_bcast_rows(in, cb_beta, i, i, 0);
        else copy_tile(in, i, 0);
        round_bf16_rne_tile(0);   // the packer truncates; ttnn.layer_norm's output rounds
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(in, KC);
}

void kernel_main() {
    binary_op_init_common(cb_x, cb_scaler, cb_p1);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    if constexpr (HAS_W) cb_wait_front(cb_gamma, KC);
    if constexpr (HAS_B) cb_wait_front(cb_beta, KC);

    ln_prologue::row_mean<LnCbs>(cb_x, cb_p1, false);         // partial sum / K
    sum_parts(cb_parts1, cb_mean);                        // mean
    cb_wait_front(cb_mean, 1);
    ln_prologue::stage<LnCbs, ln_prologue::SUB_COL, 4>(cb_x, cb_mean, cb_xc);
    cb_pop_front(cb_x, KC);
    cb_pop_front(cb_mean, 1);
    cb_wait_front(cb_xc, KC);
    ln_prologue::stage<LnCbs, ln_prologue::MUL, 4>(cb_xc, cb_xc, cb_tmp);
    ln_prologue::row_mean<LnCbs>(cb_tmp, cb_p2, true);        // partial sum (x - mean)^2 / K
    sum_parts(cb_parts2, cb_var);                         // biased variance
    ln_prologue::rstd<LnCbs>();
    ln_prologue::stage<LnCbs, ln_prologue::MUL_COL, 4>(cb_xc, cb_rstd, cb_tmp);
    cb_pop_front(cb_xc, KC);
    cb_pop_front(cb_rstd, 1);
    if constexpr (HAS_W) {
        cb_wait_front(cb_tmp, KC);
        ln_prologue::stage<LnCbs, ln_prologue::MUL_ROW, 4>(cb_tmp, cb_gamma, cb_xc);
        cb_pop_front(cb_tmp, KC);
        finish(cb_xc);
    } else {
        finish(cb_tmp);
    }
}
