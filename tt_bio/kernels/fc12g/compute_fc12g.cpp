// SPDX-License-Identifier: Apache-2.0
//
// tt-bio fc12g: silu(x @ w1) * (x @ w2) in one 2D-mcast matmul pass, for the Transition swiglu.
//
// The host runs ttnn's 2D multicast reuse matmul (tt_bio/mm2d_generic.py) over weights laid out as [w1_j | w2_j] per
// core column j, so each core's in1 block is its fc1 columns followed by its fc2 columns. This replaces only the
// compute kernel (bmm_large_block_zm_fused_bias_activation.cpp at v0.68.0); its compile-time args are the stock
// ones. Per output row h of subblocks:
//   fc1 subblock -> silu -> packed into the gate CB
//   fc2 subblock -> dest *= gate (dest reused as srcB, gate unpacked to srcA) -> packed to the output shard
// so the output is [M, N/2] and neither fc1 nor fc2 is ever written to L1 as a tensor, and no separate multiply runs.
//
// Requirements, asserted below: one K block (no spill/reload), no batch or outer blocking, two in1 subblocks per
// row (fc1 then fc2). The gate CB, in1 and the output share a data format, so no unpack/pack reconfig is needed
// between the phases.
//
// FC12G_PACK_SILU: the silu runs on the PACK thread over the fc1 dest half while MATH starts fc2 in the other half
// (the same handoff as the silu_f32 overlay of the stock kernel, tt_bio/metal_overlay.py). It needs the silu_f32
// overlay (APPROX silu = calculate_silu_f32). Without it the silu runs on MATH after the fc1 matmul.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/activations.h"
#include "experimental/circular_buffer.h"

#if defined(FC12G_PACK_SILU) && defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_silu.h"
#endif

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t batch = get_compile_time_arg_val(13);
    static_assert(num_blocks_inner_dim == 1 && num_blocks_w_dim == 1 && num_blocks_h_dim == 1 && batch == 1);
    static_assert(in1_num_subblocks == 2, "in1 block = [fc1 subblock | fc2 subblock]");
    static_assert(out_subblock_num_tiles == out_subblock_h * out_subblock_w);

    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t gate_cb_id = get_named_compile_time_arg_val("cb_intermed0");
    constexpr uint32_t n = out_subblock_num_tiles;

    experimental::CircularBuffer in0_cb(in0_cb_id);
    experimental::CircularBuffer in1_cb(in1_cb_id);
    experimental::CircularBuffer out_cb(out_cb_id);
    experimental::CircularBuffer gate_cb(gate_cb_id);

    mm_block_init(in0_cb_id, in1_cb_id, out_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
#ifdef FC12G_PACK_SILU
    PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));
#else
    silu_tile_init();
#endif

    in0_cb.wait_front(in0_block_num_tiles);
    in1_cb.wait_front(in1_block_num_tiles);

    uint32_t in0_offset = 0;
    for (uint32_t h = 0; h < in0_num_subblocks; h++) {
        // fc1: in1 columns [0, out_subblock_w)
        tile_regs_acquire();
        for (uint32_t k = 0, in0_i = in0_offset, in1_i = 0; k < in0_block_w; k++, in0_i++, in1_i += in1_block_w) {
            matmul_block(in0_cb_id, in1_cb_id, in0_i, in1_i, 0, false, out_subblock_w, out_subblock_h, in0_block_w);
        }
#ifndef FC12G_PACK_SILU
        for (uint32_t i = 0; i < n; i++) {
            silu_tile(i);
        }
#endif
        tile_regs_commit();
        gate_cb.reserve_back(n);
#ifdef FC12G_PACK_SILU
        // tile_regs_wait() that also points the SFPU at the packer's dest half, then silu before the pack
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t i = 0; i < n; i++) {
            PACK((llk_math_eltwise_unary_sfpu_silu<true, DST_ACCUM_MODE>(i)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#else
        tile_regs_wait();
#endif
        pack_tile_block(0, gate_cb_id, n);
        tile_regs_release();
        gate_cb.push_back(n);

        // fc2: in1 columns [out_subblock_w, 2 * out_subblock_w), then times the gate in dest
        tile_regs_acquire();
        for (uint32_t k = 0, in0_i = in0_offset, in1_i = out_subblock_w; k < in0_block_w;
             k++, in0_i++, in1_i += in1_block_w) {
            matmul_block(in0_cb_id, in1_cb_id, in0_i, in1_i, 0, false, out_subblock_w, out_subblock_h, in0_block_w);
        }
        binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(gate_cb_id);
        gate_cb.wait_front(n);
        for (uint32_t i = 0; i < n; i++) {
            binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(gate_cb_id, i, i);
        }
        gate_cb.pop_front(n);
        mm_block_init_short_with_dt(
            in0_cb_id, in1_cb_id, gate_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
        tile_regs_commit();
        out_cb.reserve_back(n);
        tile_regs_wait();
        pack_tile_block(0, out_cb_id, n);
        tile_regs_release();
        out_cb.push_back(n);

        in0_offset += in0_subblock_num_tiles;
    }
    in0_cb.pop_front(in0_block_num_tiles);
    in1_cb.pop_front(in1_block_num_tiles);
}
