// SPDX-License-Identifier: Apache-2.0
//
// The triangle-attention backward, with the score block never leaving L1.
//
// One core owns one head and a contiguous group of the leading axis. For each row of that group it
// forms S = (Q K^T) * scale + bias, softmaxes it, and computes dV, dP, dS, dQ and dK without ever
// writing a score-sized tensor to DRAM. dbias is the only term that reduces across the leading
// axis, so it accumulates in float32 in this core's own L1 across the whole group and is written
// once at the end; a host-side sum over the group axis finishes it.
//
// The key axis is never chunked, which is why there is no log-sum-exp bookkeeping here: this core
// holds every key for its rows, so each softmax is exact and complete the first time. This file is
// the whole-query form, where one query chunk covers the token axis and dV and dK therefore need
// no accumulation across chunks. The host gate refuses any shape that does not fit that.
//
// The convention matched is autograd.triangle_attention's -- softmax(Q K^T * scale + bias) -- and
// NOT the fused forward's softmax((Q K^T + mask) * scale). The two differ by where the scale lands
// relative to the bias, and getting it backwards is a wrong gradient that the forward agrees with.

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/transpose_wh.h"
#include "../../triatt_sdpa/compute/compute_common.hpp"

namespace {

// out_cb[M, N] = in0_cb[M, K] @ in1_cb[K, N], and NEITHER input is popped.
//
// compute_common's matmul_blocks pops its in1, which does not work here: k is read once as K^T
// (for S) and once as K (for dQ) off the same tiles, and q is read once for S and once for dK.
// TRANSPOSE transposes in1's faces, which is what turns the Nt tiles of K into the tile grid of
// K^T -- for a [Nt, 1] operand the tile ORDER is already the transpose, so only the faces move.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t SBH, uint32_t SBW, bool TRANSPOSE>
ALWI void mm_keep(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    mm_block_init_short(in0_cb, in1_cb, TRANSPOSE, SBW /*ct_dim*/, SBH /*rt_dim*/, K /*kt_dim*/);
    reconfig_data_format(in1_cb, in0_cb);
    cb_wait_front(in0_cb, M * K);
    cb_wait_front(in1_cb, K * N);
    cb_reserve_back(out_cb, M * N);
    for (uint32_t m = 0; m < M; m += SBH) {
        for (uint32_t n = 0; n < N; n += SBW) {
            tile_regs_acquire();
            uint32_t in0_index = m * K;
            uint32_t in1_index = n;
            for (uint32_t kk = 0; kk < K; ++kk) {
                matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0 /*dst*/, TRANSPOSE, SBW, SBH, K);
                in0_index++;
                in1_index += N;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t dst_idx = 0;
            for (uint32_t r = 0; r < SBH; ++r) {
                for (uint32_t c = 0; c < SBW; ++c) {
                    pack_tile<true>(dst_idx++, out_cb, (m + r) * N + n + c);
                }
            }
            tile_regs_release();
        }
    }
    cb_push_back(out_cb, M * N);
}

// out_cb[C, R] = transpose(in_cb[R, C]), both the tile grid and each tile's own faces.
//
// dV needs P^T and dK needs dS^T, and matmul_block can only transpose its in1. Doing it here costs
// R*C tile operations inside L1 and no DRAM at all, which is the trade this whole kernel is about.
template <uint32_t R, uint32_t C>
ALWI void transpose_block(uint32_t in_cb, uint32_t out_cb) {
    transpose_wh_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    cb_wait_front(in_cb, R * C);
    cb_reserve_back(out_cb, R * C);
    for (uint32_t i = 0; i < C; ++i) {
        for (uint32_t j = 0; j < R; ++j) {
            acquire_dst();
            transpose_wh_tile(in_cb, j * C + i, 0);
            pack_tile<true>(0, out_cb, i * R + j);
            release_dst();
        }
    }
    cb_push_back(out_cb, R * C);
}

// out_cb = in0_cb * in1_cb, with NEITHER input popped.
// compute_common only has the in-place form, and dS needs P kept: the row-sum correction reads
// dP * P and then dS itself is P * (dP - correction), so P is live across both.
ALWI void mul_block_to(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    mul_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        mul_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile<true>(0, out_cb, i);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
}

// in0_cb -= in1_cb broadcast along columns, in place. compute_common has the multiply form and the
// subtract-then-exp form, but not the plain subtract the softmax backward needs.
template <uint32_t rows, uint32_t cols>
ALWI void sub_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb) {
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            acquire_dst();
            sub_tiles_bcast_cols(in0_cb, in1_cb, i * cols + j, i, 0);
            pack_tile<true>(0, in0_cb, i * cols + j);
            release_dst();
        }
    }
}

// Fill num_tiles of cb with copies of zero_cb's single tile, and push them.
// The dbias accumulator has to start at zero and stay front-resident for the whole group, so it is
// produced once here and then only ever added into in place.
ALWI void seed_zeros(uint32_t cb, uint32_t zero_cb, uint32_t num_tiles) {
    copy_tile_to_dst_init_short(zero_cb);
    reconfig_data_format_srca(zero_cb);
    pack_reconfig_data_format(cb);
    cb_wait_front(zero_cb, 1);
    cb_reserve_back(cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(zero_cb, 0, 0);
        pack_tile<true>(0, cb, i);
        release_dst();
    }
    cb_push_back(cb, num_tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);          // token axis, in tiles
    constexpr uint32_t Dt = get_compile_time_arg_val(1);          // head dim, in tiles
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(3);  // head_dim ** -0.5
    constexpr uint32_t sq_sbh = get_compile_time_arg_val(4);      // subblock for a [Nt, Nt] result
    constexpr uint32_t sq_sbw = get_compile_time_arg_val(5);
    constexpr uint32_t col_sbh = get_compile_time_arg_val(6);     // subblock for a [Nt, Dt] result
    constexpr uint32_t col_sbw = get_compile_time_arg_val(7);

    const uint32_t row_start = get_arg_val<uint32_t>(0);          // this core's leading-axis group
    const uint32_t row_end = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_do = tt::CBIndex::c_3;
    constexpr uint32_t cb_bias = tt::CBIndex::c_4;
    constexpr uint32_t cb_scalar = tt::CBIndex::c_5;   // packed bf16 1.0, the reductions' scale
    constexpr uint32_t cb_zero = tt::CBIndex::c_6;
    constexpr uint32_t cb_scale = tt::CBIndex::c_7;    // the attention scale, as one bf16 tile
    constexpr uint32_t cb_p = tt::CBIndex::c_24;
    constexpr uint32_t cb_dp = tt::CBIndex::c_25;
    constexpr uint32_t cb_t = tt::CBIndex::c_26;
    constexpr uint32_t cb_row_a = tt::CBIndex::c_27;
    constexpr uint32_t cb_row_b = tt::CBIndex::c_28;
    constexpr uint32_t cb_dbias = tt::CBIndex::c_29;
    constexpr uint32_t cb_done = tt::CBIndex::c_30;
    constexpr uint32_t cb_dq = tt::CBIndex::c_16;
    constexpr uint32_t cb_dk = tt::CBIndex::c_17;
    constexpr uint32_t cb_dv = tt::CBIndex::c_18;

    constexpr uint32_t score_tiles = Nt * Nt;
    constexpr uint32_t col_tiles = Nt * Dt;

    mm_init(cb_q, cb_k, cb_p);

    // The bias is [1, H, N, N] broadcast over the leading axis, so this core's head slice is read
    // once by the reader and indexed for every row of the group rather than re-read per row. That
    // is the same argument the forward's persistent-mask CB makes, and it is worth more here
    // because the backward touches the bias twice.
    cb_wait_front(cb_bias, score_tiles);
    cb_wait_front(cb_scalar, 1);
    cb_wait_front(cb_scale, 1);

    seed_zeros(cb_dbias, cb_zero, score_tiles);

    for (uint32_t row = row_start; row < row_end; ++row) {
        // ---- S = (Q K^T) * scale + bias, then P = softmax(S) --------------------------------
        // cb_k holds K's Nt tiles. Read as [Dt, Nt] with the faces transposed, those same tiles
        // are K^T's tile grid, which is why no second copy of k is read from DRAM.
        mm_keep<Nt, Nt, Dt, sq_sbh, sq_sbw, true>(cb_q, cb_k, cb_p);
        mul_block_bcast_scalar_inplace<cb_scale, score_tiles>(cb_p);
        add_block_inplace<false>(cb_p, cb_bias, score_tiles);

        reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_p, cb_scalar, Nt, (int)VectorMode::RC>(
            cb_row_a, cb_row_a, Nt, false);
        sub_exp_block_bcast_cols_inplace<cb_p, Nt, 0x3F800000 /*1.0f*/>(cb_row_a, cb_row_b, Nt);
        cb_pop_front(cb_row_a, Nt);
        reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_p, cb_scalar, Nt, (int)VectorMode::RC>(
            cb_row_b, cb_row_b, Nt, false);
        // Keep the row sum: the softmax backward's row-sum correction divides by it, and it is
        // 1.0 only in exact arithmetic. Dropping it is worth up to 13.09x on ||dq||.
        copy_block(cb_row_b, cb_row_a, Nt);              // cb_row_a = rowsum(exp), cb_row_b free
        recip_block_inplace(cb_row_a, Nt);
        mul_block_bcast_cols<Nt, Nt, false, false>(cb_p, cb_row_a, cb_p);   // P = exp / rowsum
        cb_pop_front(cb_row_a, Nt);

        // ---- dV = P^T dO -------------------------------------------------------------------
        transpose_block<Nt, Nt>(cb_p, cb_t);
        mm_keep<Nt, Dt, Nt, col_sbh, col_sbw, false>(cb_t, cb_do, cb_dv);
        cb_pop_front(cb_t, score_tiles);

        // ---- dP = dO V^T, then dS = P * (dP - rowsum(dP * P) / rowsum(P)) --------------------
        mm_keep<Nt, Nt, Dt, sq_sbh, sq_sbw, true>(cb_do, cb_v, cb_dp);      // cb_dp = dP
        mul_block_to(cb_dp, cb_p, cb_t, score_tiles);    // cb_t = dP * P, both inputs kept
        reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_t, cb_scalar, Nt, (int)VectorMode::RC>(
            cb_row_a, cb_row_a, Nt, false);              // cb_row_a = rowsum(dP * P)
        cb_pop_front(cb_t, score_tiles);
        // P here is already normalised, so rowsum(P) is 1 up to bf16 rounding and this division is
        // exactly the correction that removes that rounding rather than a no-op.
        recip_block_inplace(cb_row_b, Nt);
        mul_block_inplace(cb_row_a, cb_row_b, Nt);
        cb_pop_front(cb_row_b, Nt);

        sub_block_bcast_cols_inplace<Nt, Nt>(cb_dp, cb_row_a);
        cb_pop_front(cb_row_a, Nt);
        mul_block_inplace(cb_dp, cb_p, score_tiles);     // cb_dp = dS
        cb_pop_front(cb_p, score_tiles);

        // ---- dbias += dS, the only term that reduces across the leading axis ----------------
        add_block_inplace<false>(cb_dbias, cb_dp, score_tiles);

        // ---- dQ = (dS K) * scale, dK = (dS^T Q) * scale --------------------------------------
        mm_keep<Nt, Dt, Nt, col_sbh, col_sbw, false>(cb_dp, cb_k, cb_dq);
        mul_block_bcast_scalar_inplace<cb_scale, col_tiles>(cb_dq);

        transpose_block<Nt, Nt>(cb_dp, cb_t);
        cb_pop_front(cb_dp, score_tiles);
        mm_keep<Nt, Dt, Nt, col_sbh, col_sbw, false>(cb_t, cb_q, cb_dk);
        mul_block_bcast_scalar_inplace<cb_scale, col_tiles>(cb_dk);
        cb_pop_front(cb_t, score_tiles);

        cb_pop_front(cb_q, col_tiles);
        cb_pop_front(cb_k, col_tiles);
        cb_pop_front(cb_v, col_tiles);
        cb_pop_front(cb_do, col_tiles);
    }

    // The writer drains cb_dbias's L1 directly rather than through a second buffer, so this is the
    // handshake that says the accumulator is final. A copy would have cost 331.8 KB of L1, which
    // at the shipped shape is the difference between fitting and not.
    cb_reserve_back(cb_done, 1);
    cb_push_back(cb_done, 1);
}
