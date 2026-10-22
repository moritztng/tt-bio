// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_ln compute, per row block (32 rows, Ct channel tiles x in bf16):
//   mean = sum_c x / H                       reduce into a float32 DEST
//   xc = x - mean, sq = xc^2                 one unpack of x: xc twice into DEST, the second squared
//   rstd = rsqrt(sum_c sq / H + eps)         reduce, + eps and rsqrt on the same DEST tile
//   y = xc * rstd * gamma + beta             gamma and beta multiplied and added on DEST (dest reuse)
// xc and sq stay float32 in L1; only y is packed bf16. Three packs a tile against the six of the
// staged form (kernels/back_ln), which is what bounds a norm that reads and writes each byte once.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t Ct = get_compile_time_arg_val(0);
constexpr uint32_t EPS = get_compile_time_arg_val(1);   // float32 bits
constexpr uint32_t ABL = get_compile_time_arg_val(2);   // 1: copy x to y (movement only). Diagnostic.
constexpr uint32_t G = 2;                               // xc tiles per DEST group (plus their squares)
constexpr uint32_t GY = 4;                              // y tiles per DEST group
constexpr uint32_t cb_in = 0, cb_gamma = 2, cb_beta = 3, cb_scaler = 4, cb_mean = 6, cb_rstd = 8,
                   cb_xc = 9, cb_sq = 10, cb_out = 16;

ALWI void row_sum(uint32_t in) {
    reduce_init<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, cb_mean);
    for (uint32_t t = 0; t < Ct; ++t) reduce_tile<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, t, 0, 0);
    reduce_uninit<true>();
}

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    binary_op_init_common(cb_in, cb_scaler, cb_out);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_gamma, Ct);
    cb_wait_front(cb_beta, Ct);
    for (uint32_t r = 0; r < num_rows; ++r) {
        cb_wait_front(cb_in, Ct);
        if constexpr (ABL & 1) {
            reconfig_data_format_srca(cb_in);
            pack_reconfig_data_format(cb_out);
            copy_tile_to_dst_init_short(cb_in);
            cb_reserve_back(cb_out, Ct);
            for (uint32_t t = 0; t < Ct; ++t) {
                tile_regs_acquire();
                copy_tile(cb_in, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
            }
            cb_push_back(cb_out, Ct);
            cb_pop_front(cb_in, Ct);
            continue;
        }
        // mean
        reconfig_data_format(cb_in, cb_scaler);
        pack_reconfig_data_format(cb_mean);
        cb_reserve_back(cb_mean, 1);
        tile_regs_acquire();
        row_sum(cb_in);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_mean);
        tile_regs_release();
        cb_push_back(cb_mean, 1);
        cb_wait_front(cb_mean, 1);

        // xc and xc^2 (cb_xc, cb_sq and cb_mean are all float32: one pack format)
        reconfig_data_format(cb_in, cb_mean);
        for (uint32_t t = 0; t < Ct; t += G) {
            sub_bcast_cols_init_short(cb_in, cb_mean);
            tile_regs_acquire();
            for (uint32_t k = 0; k < G; ++k) {
                sub_tiles_bcast_cols(cb_in, cb_mean, t + k, 0, k);
                sub_tiles_bcast_cols(cb_in, cb_mean, t + k, 0, G + k);
            }
            square_tile_init();
            for (uint32_t k = 0; k < G; ++k) square_tile(G + k);
            tile_regs_commit();
            cb_reserve_back(cb_xc, G);
            cb_reserve_back(cb_sq, G);
            tile_regs_wait();
            for (uint32_t k = 0; k < G; ++k) pack_tile(k, cb_xc);
            for (uint32_t k = 0; k < G; ++k) pack_tile(G + k, cb_sq);
            tile_regs_release();
            cb_push_back(cb_xc, G);
            cb_push_back(cb_sq, G);
        }
        cb_pop_front(cb_in, Ct);
        cb_pop_front(cb_mean, 1);

        // rstd
        cb_wait_front(cb_sq, Ct);
        reconfig_data_format(cb_sq, cb_scaler);
        cb_reserve_back(cb_rstd, 1);
        tile_regs_acquire();
        row_sum(cb_sq);
        binop_with_scalar_tile_init();
        add_unary_tile(0, EPS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_rstd);
        tile_regs_release();
        cb_push_back(cb_rstd, 1);
        cb_pop_front(cb_sq, Ct);

        // y = xc * rstd * gamma + beta
        cb_wait_front(cb_rstd, 1);
        cb_wait_front(cb_xc, Ct);
        cb_reserve_back(cb_out, Ct);
        for (uint32_t t = 0; t < Ct; t += GY) {
            tile_regs_acquire();
            reconfig_data_format(cb_xc, cb_rstd);
            mul_bcast_cols_init_short(cb_xc, cb_rstd);
            for (uint32_t k = 0; k < GY; ++k) mul_tiles_bcast_cols(cb_xc, cb_rstd, t + k, 0, k);
            reconfig_data_format_srca(cb_gamma);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_gamma);
            for (uint32_t k = 0; k < GY; ++k)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_gamma, t + k, k);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_beta);
            for (uint32_t k = 0; k < GY; ++k)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_beta, t + k, k);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            for (uint32_t k = 0; k < GY; ++k) pack_tile(k, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, Ct);
        cb_pop_front(cb_xc, Ct);
        cb_pop_front(cb_rstd, 1);
    }
}
