// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// tt-bio mm_gate: the wheel's minimal_matmul kernels (v0.68.0) with an MM_GATED epilogue; unchanged without it.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"
#if defined(MM_GATE_PACK_SFPU) && defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_silu.h"
#include "llk_math_eltwise_binary_sfpu_binop.h"
#endif

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            copy_tile(in_cb, tile_id, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}

#ifdef MM_ROPE_T
// tt-bio mm_rope: the qkv projection's bias and NeoX RoPE as the matmul's epilogue. Each row of the block is first
// biased into a bf16 row (c_9, the qkv tile today's matmul writes, bit for bit), then each q / k head (tiles a, b at
// core-local columns j0 + n < MM_ROPE_P, the host's weight permutation) is rotated with the row's cos / sin tile
// (c_7 / c_8, tile m) as qk_rope_heads' QKRH_SFPU does it, out_a = a cos - b sin, out_b = b cos + a sin, all four
// products in fp32 dest; v tiles are copied.
void rope_block(
    uint32_t in_cb, uint32_t bias_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles, uint32_t j0) {
    constexpr uint32_t cos_cb = tt::CBIndex::c_7, sin_cb = tt::CBIndex::c_8, tmp_cb = tt::CBIndex::c_9;
    constexpr uint32_t P = MM_ROPE_P;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        cb_reserve_back(tmp_cb, N_block_tiles);
        add_bcast_rows_init_short(in_cb, bias_cb);
        reconfig_data_format(in_cb, bias_cb);
        pack_reconfig_data_format(tmp_cb);
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, m * N_block_tiles + n, n, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, tmp_cb);
            tile_regs_release();
        }
        cb_push_back(tmp_cb, N_block_tiles);
        cb_wait_front(tmp_cb, N_block_tiles);
        pack_reconfig_data_format(out_cb);
        for (uint32_t n = 0; n < N_block_tiles;) {
            tile_regs_acquire();
            if (j0 + n < P) {
                reconfig_data_format(tmp_cb, cos_cb);
                mul_tiles_init(tmp_cb, cos_cb);
                mul_tiles(tmp_cb, cos_cb, n, m, 0);
                mul_tiles(tmp_cb, cos_cb, n + 1, m, 1);
                mul_tiles_init(tmp_cb, sin_cb);
                mul_tiles(tmp_cb, sin_cb, n + 1, m, 2);
                mul_tiles(tmp_cb, sin_cb, n, m, 3);
                sub_binary_tile_init();
                sub_binary_tile(0, 2, 0);
                add_binary_tile_init();
                add_binary_tile(1, 3, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, out_cb);
                pack_tile(1, out_cb);
                n += 2;
            } else {
                reconfig_data_format_srca(tmp_cb);
                copy_tile_to_dst_init_short(tmp_cb);
                copy_tile(tmp_cb, n, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, out_cb);
                n += 1;
            }
            tile_regs_release();
        }
        cb_push_back(out_cb, N_block_tiles);
        cb_pop_front(tmp_cb, N_block_tiles);
    }
}
#endif

#ifdef MM_GATED
// tt-bio mm_gate: the SwiGLU gate as the matmul's epilogue. Each row of the accumulated block holds fc1's N_block / 2
// tiles then fc2's (the weight is interleaved [W1_j | W2_j] per half block), so the row's output is
// silu(fc1) * fc2 at half width, computed from the accumulator in dest before the one bf16 pack.
void gate_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    // MM_GATE_GROUP output tiles per dest acquire: fc1's in dest [0, g), fc2's in [g, 2g), so the silu and multiply
    // SFPU inits run once a group instead of once a tile. 2 * g tiles must fit one dest half (4 under fp32 acc).
#ifndef MM_GATE_GROUP
#define MM_GATE_GROUP 2
#endif
    // MM_GATE_PROBE (benchmarks only, wrong output): 1 skips the silu, 2 also the multiply (copy and pack only).
    constexpr uint32_t g = MM_GATE_GROUP;
    const uint32_t half = N_block_tiles / 2;
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        const uint32_t row = m * N_block_tiles;
        for (uint32_t n0 = 0; n0 < half; n0 += g) {
            const uint32_t c = half - n0 < g ? half - n0 : g;
            tile_regs_acquire();
            copy_tile_to_dst_init_short(in_cb);
            for (uint32_t i = 0; i < c; i++) {
                copy_tile(in_cb, row + n0 + i, i);
                copy_tile(in_cb, row + half + n0 + i, g + i);
            }
#ifdef MM_GATE_PACK_SFPU
            // MATH only copies; PACK runs the silu and multiply on its own dest half (the fc12g / silu_f32 handoff:
            // a tile_regs_wait() that also points the SFPU at the packer's half), so MATH moves on to the next copy
            // or block while PACK works. APPROX silu is calculate_silu_f32 under the silu_f32 overlay.
            tile_regs_commit();
            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#if !defined(MM_GATE_PROBE) || MM_GATE_PROBE < 1
            PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));
            for (uint32_t i = 0; i < c; i++) {
                PACK((llk_math_eltwise_unary_sfpu_silu<true, DST_ACCUM_MODE>(i)));
            }
#endif
#if !defined(MM_GATE_PROBE) || MM_GATE_PROBE < 2
            PACK((llk_math_eltwise_binary_sfpu_binop_init<true, ckernel::BinaryOp::MUL>()));
            for (uint32_t i = 0; i < c; i++) {
                PACK((llk_math_eltwise_binary_sfpu_binop_mul<true, ckernel::BinaryOp::MUL, DST_ACCUM_MODE>(i, g + i, i)));
            }
#endif
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#else
#if !defined(MM_GATE_PROBE) || MM_GATE_PROBE < 1
            silu_tile_init();
            for (uint32_t i = 0; i < c; i++) {
                silu_tile(i);
            }
#endif
#if !defined(MM_GATE_PROBE) || MM_GATE_PROBE < 2
            mul_binary_tile_init();
            for (uint32_t i = 0; i < c; i++) {
                mul_binary_tile(i, g + i, i);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
#endif
            for (uint32_t i = 0; i < c; i++) {
                pack_tile(i, out_cb);
            }
            tile_regs_release();
        }
        cb_push_back(out_cb, half);
    }
}
#endif

// For caller: if FUSE_TERNARY defined then out_cb == in_cb
/**
 * Add bias to input block
 * Performs: output = input + bias (row broadcast)
 *
 * stream_output:
 *   - true: Pushes tiles one row at a time (for intermediate output to next stage)
 *   - false: Pushes all tiles at end (for final output)
 */
void add_bias_block(uint32_t in_cb, uint32_t bias_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    add_bcast_rows_init_short(in_cb, bias_cb);
    reconfig_data_format(in_cb, bias_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, tile_id, n, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}

#ifdef MM_RESID
// tt-bio mm_resid: the block's residual add as the epilogue, out = resid + (x W + bias). The bias is added as
// add_bias_block does it, then the residual's tile (c_5, the [M, N] block the non-writing DM kernel reads) is added
// to dest on the FPU, DG tiles at a time so each init serves several tiles. One rounding to the output format.
void add_bias_resid_block(
    uint32_t in_cb, uint32_t bias_cb, uint32_t resid_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    const uint32_t dg = N_block_tiles % 4 == 0 ? 4 : N_block_tiles % 2 == 0 ? 2 : 1;
    pack_reconfig_data_format(out_cb);
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n0 = 0; n0 < N_block_tiles; n0 += dg) {
            const uint32_t t0 = m * N_block_tiles + n0;
            tile_regs_acquire();
            add_bcast_rows_init_short(in_cb, bias_cb);
            reconfig_data_format(in_cb, bias_cb);
            for (uint32_t j = 0; j < dg; j++) {
                add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, t0 + j, n0 + j, j);
            }
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(resid_cb);
            reconfig_data_format_srcb(resid_cb);
            for (uint32_t j = 0; j < dg; j++) {
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    resid_cb, t0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < dg; j++) {
                pack_tile(j, out_cb);
            }
            tile_regs_release();
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}
#endif

void add_bias_and_addcmul_block(
    uint32_t intermediate_cb,
    uint32_t bias_cb,
    uint32_t ternary_a_cb,
    uint32_t ternary_b_cb,
    uint32_t scalar_value,
    uint32_t out_cb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles) {
    // Note: unary_bcast_tile does not work with fp32_acc_to_dest=True.
    // As a workaround, we perform addcmul through multiple LLKs calls (mul_tiles, mul_unary_tile, add_tiles_bcast).

    const uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t DST_ID = 0;
#ifdef FUSE_BIAS
    // ============================================
    // STEP 1: Add bias block
    // Read from intermediate_cb and write back to intermediate_cb
    // ============================================

    add_bcast_rows_init_short(intermediate_cb, bias_cb);
    reconfig_data_format(intermediate_cb, bias_cb);
    pack_reconfig_data_format(intermediate_cb);

    // Wait for ALL input data ONCE at the beginning
    cb_wait_front(bias_cb, N_block_tiles);

    // Unpacker waits for intermediate_cb to be ready
    cb_wait_front(intermediate_cb, out_block_num_tiles);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            uint32_t tile_id = m * N_block_tiles + n;

            tile_regs_acquire();
            add_tiles_bcast<BroadcastType::ROW>(intermediate_cb, bias_cb, tile_id, n, DST_ID);

            tile_regs_commit();

            tile_regs_wait();
            pack_tile(DST_ID, intermediate_cb);
            tile_regs_release();
        }
    }

    // Pop input and push output ONCE at the end
    // cb_wait_front(intermediate_cb, out_block_num_tiles); // Unpacker-Packer sync
    // cb_pop_front(intermediate_cb, out_block_num_tiles);
    cb_pop_front(bias_cb, N_block_tiles);

    cb_pop_front(intermediate_cb, out_block_num_tiles);

    // Restore intermediate_cb to ready (+ sync packer/unpacker)
    cb_reserve_back(intermediate_cb, out_block_num_tiles);
    cb_push_back(intermediate_cb, out_block_num_tiles);
#endif  // FUSE_BIAS

    // ============================================
    // STEP 2: Multiply by ternary_b (broadcast) and scalar
    // Read from intermediate_cb and write back to intermediate_cb
    // ============================================

    cb_wait_front(intermediate_cb, out_block_num_tiles);
    cb_wait_front(ternary_b_cb, N_block_tiles);

#ifndef TERNARY_B_IS_FLOAT32
    mul_bcast_rows_init_short(intermediate_cb, ternary_b_cb);
#else
    unary_bcast_init<BroadcastType::ROW>(ternary_b_cb, intermediate_cb);
#endif  // TERNARY_B_IS_FLOAT32

    binop_with_scalar_tile_init();
    reconfig_data_format(intermediate_cb, ternary_b_cb);
    pack_reconfig_data_format(intermediate_cb);

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
            // LLK BUG: unary_bcast gives bad values if mixing fp32_acc_to_dest=True and bfloat16 circular buffer
            // (https://github.com/tenstorrent/tt-llk/issues/1338)
            // To avoid the bug, we use:
            // - unary_bcast/mul_binary_tile for fp32 (more accurate)
            // - mul_tiles_bcast for bfloat16 (LLK bug workaround).

            // ternary_b_cb is [1, N], broadcast across M rows
            mul_tiles_bcast<BroadcastType::ROW>(intermediate_cb, ternary_b_cb, tile_id, n, DST_ID);
#else
            constexpr uint32_t TERNARY_B_DST_ID = 1;
            unary_bcast_init<BroadcastType::ROW>(ternary_b_cb, intermediate_cb);

            // ternary_b_cb is [1, N], broadcast across M rows
            unary_bcast<BroadcastType::ROW>(ternary_b_cb, n, TERNARY_B_DST_ID);

            copy_tile_to_dst_init_short(intermediate_cb);
            copy_tile(intermediate_cb, tile_id, DST_ID);

            mul_binary_tile_init();
            mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

            mul_unary_tile(DST_ID, scalar_value);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(DST_ID, intermediate_cb);

            tile_regs_release();
            tile_id++;
        }
    }

    cb_pop_front(ternary_b_cb, N_block_tiles);
    cb_pop_front(intermediate_cb, out_block_num_tiles);

    // 'refill' intermediate_cb (also synchronize packer/unpacker)
    cb_reserve_back(intermediate_cb, out_block_num_tiles);
    cb_push_back(intermediate_cb, out_block_num_tiles);

    cb_wait_front(intermediate_cb, out_block_num_tiles);

    add_tiles_init(intermediate_cb, ternary_a_cb);
    reconfig_data_format(intermediate_cb, ternary_a_cb);
    pack_reconfig_data_format(out_cb);

    tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        // Wait for one row of ternary_a tiles
        cb_wait_front(ternary_a_cb, N_block_tiles);

        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();

            // ternary_a_cb is pushed one row at a time, so use column index n
            add_tiles(intermediate_cb, ternary_a_cb, tile_id, n, DST_ID);

            tile_regs_commit();

            tile_regs_wait();
            pack_tile(DST_ID, out_cb);
            tile_regs_release();
            tile_id++;
        }

        cb_pop_front(ternary_a_cb, N_block_tiles);
        cb_push_back(out_cb, N_block_tiles);
    }

    cb_pop_front(intermediate_cb, out_block_num_tiles);
}

// Slightly modified from compute_common.hpp
void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb,
                    in1_cb,
                    in0_index,
                    in1_index,
                    dst_index,
                    false /*transpose*/,
                    subblock_w,
                    subblock_h,
                    K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();

            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile_id = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_start + w;
                    uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                    dst_index++;
                }
            }
            tile_regs_release();

            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);

#ifdef FUSE_TERNARY
    const uint32_t fused_ternary_scalar_uint = get_arg_val<uint32_t>(argidx++);
#else
    // Default value when ternary is not fused (not used, helps compiler optimize)
    constexpr uint32_t fused_ternary_scalar_uint = 0;
#endif

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

    constexpr uint32_t in2_cb = tt::CBIndex::c_4;
    constexpr uint32_t ternary_a_cb = tt::CBIndex::c_5;
    constexpr uint32_t ternary_b_cb = tt::CBIndex::c_6;

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

    mm_init(in0_cb, in1_cb, intermediate_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t M_num_subblocks = M_block_tiles / subblock_h;
    constexpr uint32_t N_num_subblocks = N_block_tiles / subblock_w;

    bool reuse_in0_block = false;

    uint32_t current_M_block_tiles = M_block_tiles;
    uint32_t current_N_block_tiles = N_block_tiles;
    uint32_t current_subblock_h = subblock_h;
    uint32_t current_subblock_w = subblock_w;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        current_M_block_tiles = m_tile_end - m_tile;
        current_subblock_h = std::min(current_M_block_tiles, subblock_h);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            current_N_block_tiles = n_tile_end - n_tile;
            current_subblock_w = std::min(current_N_block_tiles, subblock_w);

            mm_block_init_short(
                in0_cb,
                in1_cb,
                false /*transpose*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(intermediate_cb);
            // Accumulation buffer
            cb_reserve_back(intermediate_cb, out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                cb_wait_front(in0_cb, in0_block_num_tiles);
                cb_wait_front(in1_cb, in1_block_num_tiles);

                matmul_blocks(
                    in0_cb,
                    in1_cb,
                    intermediate_cb,
                    current_M_block_tiles,
                    current_N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    current_subblock_h,
                    current_subblock_w);

                if (k_block == K_num_blocks - 1) {
                    /**
                     * On next iteration we might get reuse on in0
                     *
                     */
                    if (n_block_iter < N_blocks_per_core - 1) {
                        // going to stride on N, so reuse in0
                        reuse_in0_block = true;
                    }
                }
                if (!reuse_in0_block) {
                    cb_pop_front(in0_cb, in0_block_num_tiles);
                }
                cb_pop_front(in1_cb, in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_push_back(intermediate_cb, out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

#ifdef MM_GATED
            cb_reserve_back(out_cb, out_block_num_tiles / 2);
            cb_wait_front(intermediate_cb, out_block_num_tiles);
            gate_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_pop_front(intermediate_cb, out_block_num_tiles);
#else
            cb_reserve_back(out_cb, out_block_num_tiles);
#ifndef FUSE_TERNARY
            cb_wait_front(intermediate_cb, out_block_num_tiles);
#ifndef FUSE_BIAS
            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
#else
            cb_wait_front(in2_cb, N_block_tiles);
#ifdef MM_ROPE_T
            cb_wait_front(tt::CBIndex::c_7, M_block_tiles);
            cb_wait_front(tt::CBIndex::c_8, M_block_tiles);
            rope_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles, n_tile - N_start_tile);
            cb_pop_front(tt::CBIndex::c_7, M_block_tiles);
            cb_pop_front(tt::CBIndex::c_8, M_block_tiles);
#elif defined(MM_RESID)
            cb_wait_front(tt::CBIndex::c_5, out_block_num_tiles);
            add_bias_resid_block(intermediate_cb, in2_cb, tt::CBIndex::c_5, out_cb, M_block_tiles, N_block_tiles);
            cb_pop_front(tt::CBIndex::c_5, out_block_num_tiles);
#else
            add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
#endif
            cb_pop_front(in2_cb, N_block_tiles);
#endif  // FUSE_BIAS
            cb_pop_front(intermediate_cb, out_block_num_tiles);

#else   // FUSE_TERNARY is set
            add_bias_and_addcmul_block(
                intermediate_cb,
                in2_cb,
                ternary_a_cb,
                ternary_b_cb,
                fused_ternary_scalar_uint,
                out_cb,
                M_block_tiles,
                N_block_tiles);
#endif  // FUSE_TERNARY
#endif  // MM_GATED
        }
    }
}
