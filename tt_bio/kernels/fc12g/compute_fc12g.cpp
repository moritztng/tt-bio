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
// FC12G_PRODUCTS=3 (the Boltz-2 token DiT transition, `_dit_ab_fused`): the in1 block is [w_g | w_2 | w_3] and the
// output silu(x w_g) * (x w_2) * (x w_3); the running product goes back through the gate CB between the products.
// Each product may span several subblocks (in1_num_subblocks = products x subblocks per product); subblock c of
// every product is finished before c + 1, so with one subblock row the output shard is written in row-major order.
//
// K may come in several blocks (num_blocks_inner_dim > 1), so in1 streams double-buffered behind the compute: every
// block but the last packs each subblock's partial product to cb_partial (fp32 under fp32 dest) and the next block
// reloads it, in the same (h, c, p) order; the silu and the multiplies run on the last block only. No batch or outer
// blocking (asserted). The gate CB, in1 and the output share a data format, so only the partials need a reconfig.
//
// FC12G_PACK_SILU: the silu runs on the PACK thread over the fc1 dest half while MATH starts fc2 in the other half
// (the same handoff as the silu_f32 overlay of the stock kernel, tt_bio/metal_overlay.py). It needs the silu_f32
// overlay (APPROX silu = calculate_silu_f32). Without it the silu runs on MATH after the fc1 matmul.
// FC12G_MUL_SFPU: the multiplies run on the SFPU in float32 (gate tile copied into the dest slot past the subblock,
// mul_binary_tile) instead of the FPU's dest-reuse multiply. On Wormhole the dest-to-srcB move keeps only the top 16
// bits of an fp32 dest, so the FPU route truncates every product to bf16 before multiplying: 0.0068 rel against
// float64 for the DiT transition's hidden against 0.0030 modelled for an fp32 multiply. Needs a free dest tile.
// FC12G_MATH_SILU=k (with FC12G_PACK_SILU): MATH takes the first k tiles of each fc1 subblock and PACK the rest, so
// the two threads' work per row can be balanced (PACK also packs both subblocks).

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "experimental/circular_buffer.h"

#if defined(FC12G_PACK_SILU) && defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_silu.h"
#endif

// Accumulate one subblock's in1 columns from `in1_col` over the K block into dest, after reloading its partial
// from the earlier K blocks when there are any.
FORCE_INLINE void fc12g_accumulate(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t partial_cb_id, bool reload,
                                   uint32_t in0_offset, uint32_t in1_col, uint32_t in0_block_w, uint32_t in1_block_w,
                                   uint32_t sw, uint32_t sh, uint32_t n) {
    if (reload) {
        experimental::CircularBuffer partial_cb(partial_cb_id);
        copy_tile_to_dst_init_short_with_dt(in1_cb_id, partial_cb_id);
        partial_cb.wait_front(n);
        copy_block_matmul_partials(partial_cb_id, 0, 0, n);
        partial_cb.pop_front(n);
        mm_block_init_short_with_dt(in0_cb_id, in1_cb_id, partial_cb_id, false, sw, sh, in0_block_w);
    }
    for (uint32_t k = 0, in0_i = in0_offset, in1_i = in1_col; k < in0_block_w; k++, in0_i++, in1_i += in1_block_w) {
        matmul_block(in0_cb_id, in1_cb_id, in0_i, in1_i, 0, false, sw, sh, in0_block_w);
    }
}

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
    static_assert(num_blocks_w_dim == 1 && num_blocks_h_dim == 1 && batch == 1);
#ifndef FC12G_PRODUCTS
#define FC12G_PRODUCTS 2
#endif
    constexpr uint32_t products = FC12G_PRODUCTS;
    static_assert(products >= 2 && in1_num_subblocks % products == 0, "in1 block = [fc1 | fc2 (| fc3)]");
    constexpr uint32_t cols = in1_num_subblocks / products;        // subblocks per product
    constexpr uint32_t product_w = cols * out_subblock_w;          // in1 columns per product
    static_assert(out_subblock_num_tiles == out_subblock_h * out_subblock_w);

    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t gate_cb_id = get_named_compile_time_arg_val("cb_intermed0");
    constexpr uint32_t partial_cb_id = get_named_compile_time_arg_val("cb_partial");
    constexpr uint32_t n = out_subblock_num_tiles;
    // MATH_FIDELITY exists only in the math thread's build; the other threads never use mul_fidelity.
#if defined(FC12G_MUL_HIFI4) || !defined(TRISC_MATH)
    constexpr auto mul_fidelity = MathFidelity::HiFi4;
#else
    constexpr auto mul_fidelity = static_cast<MathFidelity>(MATH_FIDELITY);
#endif

    experimental::CircularBuffer in0_cb(in0_cb_id);
    experimental::CircularBuffer in1_cb(in1_cb_id);
    experimental::CircularBuffer out_cb(out_cb_id);
    experimental::CircularBuffer gate_cb(gate_cb_id);
    experimental::CircularBuffer partial_cb(partial_cb_id);

    mm_block_init(in0_cb_id, in1_cb_id, out_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
#ifdef FC12G_PACK_SILU
    PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));
#ifndef FC12G_MATH_SILU
#define FC12G_MATH_SILU 0
#endif
    constexpr uint32_t math_silu = FC12G_MATH_SILU;
#else
    constexpr uint32_t math_silu = out_subblock_num_tiles;
#endif
    static_assert(math_silu <= out_subblock_num_tiles);
    if constexpr (math_silu > 0) {
        silu_tile_init();
    }

    for (uint32_t b = 0; b < num_blocks_inner_dim; b++) {
        const bool reload = b > 0;
        in0_cb.wait_front(in0_block_num_tiles);
        in1_cb.wait_front(in1_block_num_tiles);
        if (b + 1 < num_blocks_inner_dim) {
            // a K block before the last: every subblock of every product to the partials, nothing else
            PACK((pack_reconfig_data_format(partial_cb_id)));
            for (uint32_t h = 0, in0_offset = 0; h < in0_num_subblocks; h++, in0_offset += in0_subblock_num_tiles) {
                for (uint32_t c = 0; c < cols; c++) {
                    for (uint32_t p = 0; p < products; p++) {
                        tile_regs_acquire();
                        fc12g_accumulate(in0_cb_id, in1_cb_id, partial_cb_id, reload, in0_offset,
                                         p * product_w + c * out_subblock_w, in0_block_w, in1_block_w, out_subblock_w,
                                         out_subblock_h, n);
                        tile_regs_commit();
                        partial_cb.reserve_back(n);
                        tile_regs_wait();
                        pack_tile_block(0, partial_cb_id, n);
                        tile_regs_release();
                        partial_cb.push_back(n);
                    }
                }
            }
            PACK((pack_reconfig_data_format(out_cb_id)));
            in0_cb.pop_front(in0_block_num_tiles);
            in1_cb.pop_front(in1_block_num_tiles);
            continue;
        }

        uint32_t in0_offset = 0;
        for (uint32_t h = 0; h < in0_num_subblocks; h++) {
            for (uint32_t c = 0; c < cols; c++) {
                // the gate: in1 columns [c * out_subblock_w, (c + 1) * out_subblock_w), silu, into the gate CB
                tile_regs_acquire();
                fc12g_accumulate(in0_cb_id, in1_cb_id, partial_cb_id, reload, in0_offset, c * out_subblock_w,
                                 in0_block_w, in1_block_w, out_subblock_w, out_subblock_h, n);
                for (uint32_t i = 0; i < math_silu; i++) {
                    silu_tile(i);
                }
                tile_regs_commit();
                gate_cb.reserve_back(n);
#ifdef FC12G_PACK_SILU
                // tile_regs_wait() that also points the SFPU at the packer's dest half, then silu before the pack
                PACK(TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK),
                                 p_stall::STALL_ON_ZERO));
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                for (uint32_t i = math_silu; i < n; i++) {
                    PACK((llk_math_eltwise_unary_sfpu_silu<true, DST_ACCUM_MODE>(i)));
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#else
                tile_regs_wait();
#endif
                pack_tile_block(0, gate_cb_id, n);
                tile_regs_release();
                gate_cb.push_back(n);

                // each further product: its columns, times the running product in dest; the last one to the output
                for (uint32_t p = 1; p < products; p++) {
                    tile_regs_acquire();
                    fc12g_accumulate(in0_cb_id, in1_cb_id, partial_cb_id, reload, in0_offset,
                                     p * product_w + c * out_subblock_w, in0_block_w, in1_block_w, out_subblock_w,
                                     out_subblock_h, n);
#ifdef FC12G_MUL_SFPU
                    static_assert(n < 4, "FC12G_MUL_SFPU needs a dest tile past the subblock in an fp32 dest half");
                    gate_cb.wait_front(n);
                    for (uint32_t i = 0; i < n; i++) {
                        copy_tile_to_dst_init_short_with_dt(in1_cb_id, gate_cb_id);
                        copy_tile(gate_cb_id, i, n);
                        mul_binary_tile_init();
                        mul_binary_tile(i, n, i);
                    }
                    gate_cb.pop_front(n);
#else
                    // binary_dest_reuse_tiles_init/_tiles<ELWMUL, DEST_TO_SRCB>, with the multiply's fidelity a choice
                    // (FC12G_MUL_HIFI4) rather than the matmul's.
                    UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        false, false, gate_cb_id)));
                    MATH((llk_math_eltwise_binary_init<ELWMUL, NONE, mul_fidelity,
                                                       EltwiseBinaryReuseDestType::DEST_TO_SRCB>(false)));
                    gate_cb.wait_front(n);
                    for (uint32_t i = 0; i < n; i++) {
                        UNPACK((llk_unpack_A<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                            gate_cb_id, i)));
                        MATH((llk_math_eltwise_binary<ELWMUL, NONE, DST_ACCUM_MODE, mul_fidelity,
                                                      EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                            gate_cb_id, gate_cb_id, i, true)));
                    }
                    gate_cb.pop_front(n);
#endif
                    mm_block_init_short_with_dt(
                        in0_cb_id, in1_cb_id, gate_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
                    tile_regs_commit();
                    const uint32_t dst_cb_id = p + 1 == products ? out_cb_id : gate_cb_id;
                    experimental::CircularBuffer& dst_cb = p + 1 == products ? out_cb : gate_cb;
                    dst_cb.reserve_back(n);
                    tile_regs_wait();
                    pack_tile_block(0, dst_cb_id, n);
                    tile_regs_release();
                    dst_cb.push_back(n);
                }
            }
            in0_offset += in0_subblock_num_tiles;
        }
        in0_cb.pop_front(in0_block_num_tiles);
        in1_cb.pop_front(in1_block_num_tiles);
    }
}
