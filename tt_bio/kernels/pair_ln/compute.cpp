// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pair layer norm over the last dim, one tile-row (32 rows x Wt tiles) at a time, all in L1:
//   mean = mean(x); xc = x - mean; rstd = rsqrt(mean(xc^2) + eps); out = xc * rstd * gamma + beta
// Two-pass statistics as in kernels/lnbw; xc is a float32 CB and the reductions accumulate in a
// float32 DEST. gamma and beta arrive as float32 tiles with all 32 rows equal, so the affine step
// runs straight out of DEST (`binary_dest_reuse_tiles`) and the output is packed once, in the
// output CB's format (bfloat16, or bfloat8_b when the caller asks for it).
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t DB = get_compile_time_arg_val(1);   // tiles per DEST batch, divides Wt
constexpr uint32_t cb_x = 0, cb_gamma = 1, cb_beta = 2, cb_scaler = 3, cb_eps = 4, cb_mean = 5,
                   cb_xc = 6, cb_sq = 7, cb_var = 8, cb_rstd = 9, cb_out = 16;

// Sum of each row of Wt tiles of `in`, times the 1/K scaler, into one tile of `out`.
ALWI void row_mean(uint32_t in, uint32_t out) {
    reconfig_data_format(in, cb_scaler);
    pack_reconfig_data_format(out);
    cb_wait_front(in, Wt);
    cb_reserve_back(out, 1);
    tile_regs_acquire();
    reduce_init<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, out);
    for (uint32_t i = 0; i < Wt; ++i) reduce_tile<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, i, 0, 0);
    reduce_uninit<true>();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
}

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    binary_op_init_common(cb_x, cb_scaler, cb_mean);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_gamma, Wt);
    cb_wait_front(cb_beta, Wt);
    for (uint32_t r = 0; r < num_rows; ++r) {
        row_mean(cb_x, cb_mean);
        cb_wait_front(cb_mean, 1);
        // xc = x - mean
        reconfig_data_format(cb_x, cb_mean);
        pack_reconfig_data_format(cb_xc);
        sub_bcast_cols_init_short(cb_x, cb_mean);
        cb_reserve_back(cb_xc, Wt);
        for (uint32_t i = 0; i < Wt; i += DB) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < DB; ++j) sub_tiles_bcast_cols(cb_x, cb_mean, i + j, 0, j);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < DB; ++j) pack_tile(j, cb_xc);
            tile_regs_release();
        }
        cb_push_back(cb_xc, Wt);
        cb_pop_front(cb_x, Wt);
        cb_pop_front(cb_mean, 1);
        // xc^2
        cb_wait_front(cb_xc, Wt);
        reconfig_data_format(cb_xc, cb_xc);
        pack_reconfig_data_format(cb_sq);
        mul_tiles_init(cb_xc, cb_xc);
        cb_reserve_back(cb_sq, Wt);
        for (uint32_t i = 0; i < Wt; i += DB) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < DB; ++j) mul_tiles(cb_xc, cb_xc, i + j, i + j, j);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < DB; ++j) pack_tile(j, cb_sq);
            tile_regs_release();
        }
        cb_push_back(cb_sq, Wt);
        row_mean(cb_sq, cb_var);
        cb_pop_front(cb_sq, Wt);
        // rstd = rsqrt(var + eps)
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
        // out = xc * rstd * gamma + beta, one pack
        cb_wait_front(cb_rstd, 1);
        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, Wt);
        for (uint32_t i = 0; i < Wt; i += DB) {
            tile_regs_acquire();
            reconfig_data_format(cb_xc, cb_rstd);
            mul_bcast_cols_init_short(cb_xc, cb_rstd);
            for (uint32_t j = 0; j < DB; ++j) mul_tiles_bcast_cols(cb_xc, cb_rstd, i + j, 0, j);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma);
            reconfig_data_format_srcb(cb_gamma);
            for (uint32_t j = 0; j < DB; ++j)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_gamma, i + j, j);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_beta);
            for (uint32_t j = 0; j < DB; ++j)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_beta, i + j, j);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < DB; ++j) pack_tile(j, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, Wt);
        cb_pop_front(cb_xc, Wt);
        cb_pop_front(cb_rstd, 1);
    }
}
