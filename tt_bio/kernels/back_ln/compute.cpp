// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// back_ln compute, per output row block (rows j, Ct channel tiles):
//   x = [prev tiles as they are | transpose_wh of each gathered chunk tile]
//   y = (x - mean(x)) * rsqrt(mean((x - mean)^2) + eps) * gamma + beta,  mean over the H channels
// The statistics are kernels/lnbw/compute_lnbw.cpp's: float32 intermediate CBs, reductions in a
// float32 DEST.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t Ct = get_compile_time_arg_val(0);
constexpr uint32_t Cp = get_compile_time_arg_val(1);
constexpr uint32_t cb_in = 0, cb_prev = 1, cb_gamma = 2, cb_beta = 3, cb_scaler = 4, cb_eps = 5,
                   cb_mean = 6, cb_var = 7, cb_rstd = 8, cb_x = 9, cb_xc = 10, cb_tmp = 11, cb_out = 16;

ALWI void row_mean(uint32_t in, uint32_t out, bool pop) {
    reconfig_data_format(in, cb_scaler);
    pack_reconfig_data_format(out);
    cb_wait_front(in, Ct);
    cb_reserve_back(out, 1);
    tile_regs_acquire();
    reduce_init<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, out);
    for (uint32_t i = 0; i < Ct; ++i) reduce_tile<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, i, 0, 0);
    reduce_uninit<true>();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    if (pop) cb_pop_front(in, Ct);
}

enum Op { SUB_COL, MUL_COL, MUL_ROW, ADD_ROW, MUL };

// out[i] = a[i] (op) b[0 or i], Ct tiles. Neither input is popped.
template <Op OP>
ALWI void stage(uint32_t a, uint32_t b, uint32_t out) {
    reconfig_data_format(a, b);
    pack_reconfig_data_format(out);
    if constexpr (OP == SUB_COL) sub_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_COL) mul_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_ROW) mul_bcast_rows_init_short(a, b);
    if constexpr (OP == ADD_ROW) add_bcast_rows_init_short(a, b);
    if constexpr (OP == MUL) mul_tiles_init(a, b);
    cb_reserve_back(out, Ct);
    for (uint32_t i = 0; i < Ct; ++i) {
        tile_regs_acquire();
        if constexpr (OP == SUB_COL) sub_tiles_bcast_cols(a, b, i, 0, 0);
        if constexpr (OP == MUL_COL) mul_tiles_bcast_cols(a, b, i, 0, 0);
        if constexpr (OP == MUL_ROW) mul_tiles_bcast_rows(a, b, i, i, 0);
        if constexpr (OP == ADD_ROW) add_tiles_bcast_rows(a, b, i, i, 0);
        if constexpr (OP == MUL) mul_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out);
        tile_regs_release();
    }
    cb_push_back(out, Ct);
}

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    binary_op_init_common(cb_x, cb_scaler, cb_mean);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_gamma, Ct);
    cb_wait_front(cb_beta, Ct);
    for (uint32_t r = 0; r < num_rows; ++r) {
        // x: the earlier channels copied, the chunk's transposed, in channel order
        pack_reconfig_data_format(cb_x);
        cb_reserve_back(cb_x, Ct);
        if constexpr (Cp > 0) {
            reconfig_data_format_srca(cb_prev);
            copy_tile_to_dst_init_short(cb_prev);
            cb_wait_front(cb_prev, Cp);
            for (uint32_t t = 0; t < Cp; ++t) {
                tile_regs_acquire();
                copy_tile(cb_prev, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_x);
                tile_regs_release();
            }
            cb_pop_front(cb_prev, Cp);
        }
        reconfig_data_format_srca(cb_in);
        transpose_wh_init_short(cb_in);
        for (uint32_t t = Cp; t < Ct; ++t) {
            cb_wait_front(cb_in, 1);
            tile_regs_acquire();
            transpose_wh_tile(cb_in, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_x);
            tile_regs_release();
            cb_pop_front(cb_in, 1);
        }
        cb_push_back(cb_x, Ct);

        row_mean(cb_x, cb_mean, false);                       // mean
        cb_wait_front(cb_mean, 1);
        stage<SUB_COL>(cb_x, cb_mean, cb_xc);                 // xc
        cb_pop_front(cb_x, Ct);
        cb_pop_front(cb_mean, 1);
        cb_wait_front(cb_xc, Ct);
        stage<MUL>(cb_xc, cb_xc, cb_tmp);                     // xc^2
        row_mean(cb_tmp, cb_var, true);                       // var
        cb_wait_front(cb_var, 1);
        reconfig_data_format(cb_var, cb_eps);
        pack_reconfig_data_format(cb_rstd);
        add_tiles_init(cb_var, cb_eps);
        cb_reserve_back(cb_rstd, 1);
        tile_regs_acquire();
        add_tiles(cb_var, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_rstd);
        tile_regs_release();
        cb_push_back(cb_rstd, 1);
        cb_pop_front(cb_var, 1);
        cb_wait_front(cb_rstd, 1);
        stage<MUL_COL>(cb_xc, cb_rstd, cb_tmp);               // norm
        cb_pop_front(cb_xc, Ct);
        cb_pop_front(cb_rstd, 1);
        cb_wait_front(cb_tmp, Ct);
        stage<MUL_ROW>(cb_tmp, cb_gamma, cb_xc);              // norm * gamma
        cb_pop_front(cb_tmp, Ct);
        cb_wait_front(cb_xc, Ct);
        stage<ADD_ROW>(cb_xc, cb_beta, cb_out);               // + beta
        cb_pop_front(cb_xc, Ct);
    }
}
