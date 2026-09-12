// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DERIVED from tt_bio/kernels/trimul_tail/compute.cpp, which
// tt_bio/kernels/trimul_tail/patch_trimul_tail.py generated from the installed wheel's
// minimal_matmul kernels. Two changes, both named below: the fused activation is silu and it is
// applied on pass 1's fp32 accumulator (where ttnn.linear(activation="silu") applies it), and the
// epilogue is a plain product instead of a sigmoid gate.
//
//   production   x_1 = silu(x_norm @ fc1) ; x_2 = x_norm @ fc2 ; multiply_(x_1, x_2)
//   here         one kernel over both (activation, weight) pairs, product in the epilogue
//
// Pass 0 is the un-activated projection, pass 1 is the silu'd one, matching how the host binds
// (xa, wa) to pass 0 and (xb, wb) to pass 1.

#ifndef TRIMUL_TAIL_PASSES
#define TRIMUL_TAIL_PASSES 2
#endif
// How the product is rounded to bf16 before it is packed: 0 leaves it to the packer (which
// breaks ties away from zero under fp32 DST), 1 uses SFPSTOCHRND round-to-nearest-even, 2 does
// the same rounding with integer arithmetic. Only 0 is known-wrong; 1 and 2 are the candidates.
#ifndef TRIMUL_TAIL_ROUND
#define TRIMUL_TAIL_ROUND 2
#endif
// Diagnostic only: drop the gate so the multiply can be scored on its own. Never set in production.
#ifndef TRIMUL_TAIL_SKIP_SIGMOID
#define TRIMUL_TAIL_SKIP_SIGMOID 0
#endif
// How many output tiles the epilogue product folds per DST acquire. The kernel this was derived
// from takes one tile per acquire, so every output tile pays a full math/pack barrier. Each folded
// tile needs two DST slots (both operands are copied in), so under fp32 DST half sync (4 tiles) the
// ceiling is 2.
#ifndef TRIMUL_TAIL_MUL_BATCH
#define TRIMUL_TAIL_MUL_BATCH 1
#endif
// How the epilogue product reads its two operands.
//   0  two `copy_tile`s into DST, then the SFPU product. What this kernel was derived with: three
//      math passes an output tile, and measured at ~4x what `ttnn.multiply_` costs over the same
//      tiles while moving a quarter of the bytes.
//   1  `mul_tiles`, which unpacks both operands straight into the FPU in one pass, the way ttnn's
//      own binary does. One DST slot an output tile instead of two, so MUL_BATCH can go to 4 under
//      fp32 half sync. The FPU product rounds where the SFPU one does not, so this is scored on
//      PCC, not assumed equivalent.
//   2  DIAGNOSTIC ONLY, computes the WRONG answer: pack pass 0 and drop the product entirely, so
//      the epilogue's cost can be read off directly instead of inferred by subtraction. Never set
//      in production; the PCC leg in perf/b2z2_fusion/epilogue_mode.py rejects it on sight.
#ifndef TRIMUL_TAIL_MUL_MODE
#define TRIMUL_TAIL_MUL_MODE 0
#endif
// How many tiles `copy_block` folds per DST acquire as it takes the fp32 accumulator to bf16.
// The kernel this was derived from takes one tile per acquire, so the packer cannot drain tile i
// while the SFPU works on i+1. Each tile needs one DST slot, so under fp32 half sync the ceiling
// is 4. Bit-exact in every value: same tiles, same op, same pack order, only the barrier moves.
#ifndef TRIMUL_TAIL_COPY_BATCH
#define TRIMUL_TAIL_COPY_BATCH 1
#endif
// Where `silu_tile_init()` is issued. 0 = once per tile (what this kernel was derived with,
// because the epilogue's `mul_binary_tile_init()` clobbers a kernel-top init between blocks),
// 1 = once per `copy_block` call, which is still after every epilogue and so still valid.
// Config only: the SFPU function, its operand and its rounding are untouched.
#ifndef TRIMUL_TAIL_SILU_HOIST
#define TRIMUL_TAIL_SILU_HOIST 0
#endif
// Which activation pass_1's `copy_block` runs over the fp32 accumulator.
//   0  DIAGNOSTIC ONLY, computes the WRONG answer: no activation at all. It exists to price the
//      activation against the pass that carries it, and it is what found that the activation is
//      67.7 % of this kernel on WH.
//   1  `silu_tile()`, the LLK silu `ttnn.linear(activation="silu")` runs. On Wormhole that is
//      `abs`, a predicated piecewise-linear branch, a POLYVAL5, a predicated reflection and a
//      multiply -- about twenty SFPU instructions a vector, none of them unrolled.
//   2  DIAGNOSTIC ONLY, wrong answer: the kernel's own `round_bf16_tile`, a seven-instruction
//      hand-written SFPU pass, in silu's place. It measures what ANY SFPU pass over these tiles
//      costs, so silu's instruction count can be separated from the pass itself.
//   3  DIAGNOSTIC: `x * sigmoid_tile(x)` through two DST slots. It is BIT-IDENTICAL to 1, which
//      is how this kernel learned that the LLK silu IS `x * sigmoid(x)` over the same sigmoid.
//      Kept because that identity is the evidence for 4.
//   4  the same silu at bf16 accuracy instead of fp32 accuracy: `calculate_silu<false>`, which
//      takes `_sfpu_exp_21f_bf16_` (~1 ULP on bf16) and a one-iteration reciprocal where the
//      fp32 branch takes `_sfpu_exp_accurate_` and a two-iteration one. The kernel runs fp32 dest
//      acc for the MATMUL, so the LLK picks the fp32 branch for the activation as well -- and the
//      instruction after it packs the result to bf16, so that accuracy is discarded before it is
//      ever stored. A numerics change of at most one bf16 ULP; scored, never assumed.
//   5  `x * sigmoid(x)` with the SFPU's 6-entry hardware LUT sigmoid (APPROXIMATION_MODE), the
//      cheapest sigmoid on the part. A coarser approximation than 4; scored.
//   6  accurate exp, ONE Newton reciprocal step, packer rounding. Half of the step from 1 to 4.
//   7  bf16 exp, TWO Newton reciprocal steps, packer rounding. The other half.
//   8  bf16 exp, ONE Newton step, packer rounding -- 4 in every respect EXCEPT the extra
//      `float_to_fp16b` the LLK's bf16 branch applies to the result, so 8 against 4 isolates the
//      rounding mode from the arithmetic. 1, 6, 7 and 8 are the four corners of (exp, reciprocal).
#ifndef TRIMUL_TAIL_SILU
#define TRIMUL_TAIL_SILU 1
#endif
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

// ---------------------------------------------------------------- TRIMUL_TAIL: the gate epilogue
#include "api/compute/eltwise_binary_sfpu.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_silu.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_sigmoid.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
// Round DST to bf16 in the SFPU with the SAME rounding the 16-bit DST store uses, so the pack that
// follows cannot round again. Under fp32 DST the packer breaks ties away from zero where ttnn's
// binary_ng breaks them to even; E6 measured that as 0.91 % of elements at a 1.85 % tie rate.
template <int ITERATIONS = 8>
inline void _round_bf16_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
#if TRIMUL_TAIL_ROUND == 2
        // The same round-to-nearest-even by hand, in case the SFPSTOCHRND path does not land
        // where reinterpret<vFloat> expects it.
        sfpi::vUInt u = sfpi::reinterpret<sfpi::vUInt>(v);
        sfpi::vUInt lsb = (u >> 16) & 1;
        u = u + 0x7FFF;
        u = u + lsb;
        u = (u >> 16) << 16;
        sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(u);
#else
        sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(
            sfpi::float_to_fp16b(v, sfpi::RoundMode::NearestEven));
#endif
        sfpi::dst_reg++;
    }
}


// silu with the exp and the Newton reciprocal chosen INDEPENDENTLY. The LLK's own cheap branch
// (`calculate_silu<false>`, TRIMUL_TAIL_SILU 4) changes three things at once against its fp32 one:
// `_sfpu_exp_21f_bf16_` for `_sfpu_exp_accurate_`, one Newton step for two, AND a final
// `float_to_fp16b` on the result that the fp32 branch does not apply. The last one is not an
// accuracy improvement, it is a change of ROUNDING MODE: it rounds to nearest-even where the
// packer that immediately follows breaks ties away from zero. These arms take one axis at a time
// and leave the rounding to the packer in every case, so each differs from `silu_tile()` in
// exactly what its name says.
template <bool ACCURATE_EXP, int RECIP_ITERS, int ITERATIONS = 8>
inline void _silu_split_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat exp_neg_x;
        if constexpr (ACCURATE_EXP) {
            exp_neg_x = _sfpu_exp_accurate_<true>(-x);
        } else {
            exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
        }
        sfpi::dst_reg[0] = x * _sfpu_reciprocal_<RECIP_ITERS>(sfpi::vConst1 + exp_neg_x);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_MATH

ALWI void round_bf16_tile(uint32_t idst) {
    // `_llk_math_eltwise_unary_sfpu_params_` is a global template, not a member of `ckernel`.
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_round_bf16_<8>, idst, (int)VectorMode::RC)));
}

// The activation `copy_block` applies, chosen at compile time. Kept as one inline so the init and
// the per-tile call cannot drift apart.
// The LLK silu at bf16 accuracy. `silu_tile()` hard-wires `DST_ACCUM_MODE`, which this kernel sets
// for the matmul accumulator, so it cannot be asked for the cheap branch through the compute API.
ALWI void silu_bf16_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::calculate_silu<false, 8>, idst, (int)VectorMode::RC)));
}

ALWI void silu_accurate_exp_1recip_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_silu_split_<true, 1, 8>, idst, (int)VectorMode::RC)));
}

ALWI void silu_bf16_exp_2recip_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_silu_split_<false, 2, 8>, idst, (int)VectorMode::RC)));
}

ALWI void silu_bf16_exp_1recip_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_silu_split_<false, 1, 8>, idst, (int)VectorMode::RC)));
}

ALWI void sigmoid_appx_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<true>(
        ckernel::sfpu::calculate_sigmoid<true, false, 8>, idst, (int)VectorMode::RC)));
}

ALWI void tail_activation_init() {
#if TRIMUL_TAIL_SILU == 1 || TRIMUL_TAIL_SILU == 4 || TRIMUL_TAIL_SILU >= 6
    // every split arm uses the same non-approximate reciprocal table this init loads.
    silu_tile_init();
#elif TRIMUL_TAIL_SILU == 3
    sigmoid_tile_init();
#elif TRIMUL_TAIL_SILU == 5
    MATH((llk_math_eltwise_unary_sfpu_sigmoid_init<true>()));
#endif
}

ALWI void tail_activation_tile(uint32_t idst) {
#if TRIMUL_TAIL_SILU == 1
    silu_tile(idst);
#elif TRIMUL_TAIL_SILU == 2
    round_bf16_tile(idst);
#elif TRIMUL_TAIL_SILU == 4
    silu_bf16_tile(idst);
#elif TRIMUL_TAIL_SILU == 6
    silu_accurate_exp_1recip_tile(idst);
#elif TRIMUL_TAIL_SILU == 7
    silu_bf16_exp_2recip_tile(idst);
#elif TRIMUL_TAIL_SILU == 8
    silu_bf16_exp_1recip_tile(idst);
#endif
}

// out = p * g, tile by tile, with production's rounding points. `g` already carries silu and is
// already bf16 (pass 1's copy_block did both), so there is no second activation here and no
// intermediate CB: production's `ttnn.multiply_(x_1, x_2)` unpacks two bf16 operands and packs one
// bf16 result, and so does this. The multiply is the SFPU one because the FPU's `mul_tiles`
// truncates the product.
void mul_block(uint32_t p_cb, uint32_t g_cb, uint32_t out_cb, uint32_t block_num_tiles) {
    constexpr uint32_t B = TRIMUL_TAIL_MUL_BATCH;
#if TRIMUL_TAIL_MUL_MODE == 2
    copy_tile_to_dst_init_short(p_cb);
    reconfig_data_format_srca(p_cb);
    pack_reconfig_data_format(out_cb);
    for (uint32_t t = 0; t < block_num_tiles; t += B) {
        const uint32_t n = (block_num_tiles - t < B) ? (block_num_tiles - t) : B;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; i++) {
            copy_tile(p_cb, t + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
    }
    cb_push_back(out_cb, block_num_tiles);
    return;
#endif
#if TRIMUL_TAIL_MUL_MODE == 1
    // A full binary init, not a short one: the pass loop above left the hardware configured for
    // matmul. It is safe to re-init here because every pass re-issues `mm_block_init_short` and
    // both `reconfig_data_format` calls before it touches the FPU again.
    binary_op_init_common(p_cb, g_cb, out_cb);
    mul_tiles_init(p_cb, g_cb);
    for (uint32_t t = 0; t < block_num_tiles; t += B) {
        const uint32_t n = (block_num_tiles - t < B) ? (block_num_tiles - t) : B;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; i++) {
            mul_tiles(p_cb, g_cb, t + i, t + i, i);
        }
#if TRIMUL_TAIL_ROUND != 0
        for (uint32_t i = 0; i < n; i++) {
            round_bf16_tile(i);
        }
#endif
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
    }
    cb_push_back(out_cb, block_num_tiles);
    return;
#endif
    for (uint32_t t = 0; t < block_num_tiles; t += B) {
        const uint32_t n = (block_num_tiles - t < B) ? (block_num_tiles - t) : B;
        tile_regs_acquire();
        copy_tile_to_dst_init_short(p_cb);
        reconfig_data_format_srca(p_cb);
        pack_reconfig_data_format(out_cb);
        for (uint32_t i = 0; i < n; i++) {
            copy_tile(p_cb, t + i, 2 * i);
        }
        copy_tile_to_dst_init_short(g_cb);
        for (uint32_t i = 0; i < n; i++) {
            copy_tile(g_cb, t + i, 2 * i + 1);
        }
        mul_binary_tile_init();
        for (uint32_t i = 0; i < n; i++) {
            mul_binary_tile(2 * i, 2 * i + 1, 2 * i);
#if TRIMUL_TAIL_ROUND != 0
            round_bf16_tile(2 * i);
#endif
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(2 * i, out_cb);
        }
        tile_regs_release();
    }
    cb_push_back(out_cb, block_num_tiles);
}


#if TRIMUL_TAIL_SILU == 3 || TRIMUL_TAIL_SILU == 5
// silu(x) = x * sigmoid(x), with sigmoid taken from the SFPU's 6-entry hardware LUT instead of the
// LLK silu's `abs` + two predicated regions + POLYVAL5. `lut2`'s 6-entry table mode is not
// reachable from a user kernel on this sfpi (`mod1` mask 0xc rejects it), so this composes the
// same hardware path out of the compute API: one extra `copy_tile`, `sigmoid_tile`, one FPU
// multiply. Two DST slots a tile, so COPY_BATCH is capped at 2 under fp32 half sync.
void copy_block_silu_lut(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles,
                         uint32_t N_block_tiles) {
    constexpr uint32_t G = (TRIMUL_TAIL_COPY_BATCH > 2) ? 2 : TRIMUL_TAIL_COPY_BATCH;
    const uint32_t total = M_block_tiles * N_block_tiles;
    uint32_t pushed = 0;
    for (uint32_t t = 0; t < total; t += G) {
        const uint32_t n = (total - t < G) ? (total - t) : G;
        tile_regs_acquire();
        copy_tile_to_dst_init_short(in_cb);
        reconfig_data_format_srca(in_cb);
        pack_reconfig_data_format(out_cb);
        for (uint32_t i = 0; i < n; i++) {
            copy_tile(in_cb, t + i, 2 * i);
            copy_tile(in_cb, t + i, 2 * i + 1);
        }
        tail_activation_init();
        for (uint32_t i = 0; i < n; i++) {
#if TRIMUL_TAIL_SILU == 5
            sigmoid_appx_tile(2 * i + 1);
#else
            sigmoid_tile(2 * i + 1);
#endif
        }
        mul_binary_tile_init();
        for (uint32_t i = 0; i < n; i++) {
            mul_binary_tile(2 * i, 2 * i + 1, 2 * i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(2 * i, out_cb);
        }
        tile_regs_release();
        while (pushed + N_block_tiles <= t + n) {
            cb_push_back(out_cb, N_block_tiles);
            pushed += N_block_tiles;
        }
    }
}
#endif

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles,
                bool apply_silu = false) {
#if TRIMUL_TAIL_SILU == 3 || TRIMUL_TAIL_SILU == 5
    if (apply_silu) {
        copy_block_silu_lut(in_cb, out_cb, M_block_tiles, N_block_tiles);
        return;
    }
#endif
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
#if TRIMUL_TAIL_SILU && TRIMUL_TAIL_SILU_HOIST
    // One SFPU init a block instead of one a tile. The epilogue's `mul_binary_tile_init()` runs
    // between blocks, not between tiles, so a per-block init is still live for every tile it
    // covers -- which is the only reason the per-tile form existed.
    if (apply_silu) {
        tail_activation_init();
    }
#endif

    constexpr uint32_t G = TRIMUL_TAIL_COPY_BATCH;
    const uint32_t total = M_block_tiles * N_block_tiles;
    uint32_t pushed = 0;
    for (uint32_t t = 0; t < total; t += G) {
        const uint32_t n = (total - t < G) ? (total - t) : G;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; i++) {
            copy_tile(in_cb, t + i, i);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            // The fused activation goes here and nowhere else: this is the copy that takes the
            // fp32 accumulator to bf16, so silu sees the same fp32 value and the same single
            // rounding that `ttnn.linear(activation="silu")` gives it.
#if TRIMUL_TAIL_SILU
            if (apply_silu) {
#if !TRIMUL_TAIL_SILU_HOIST
                tail_activation_init();
#endif
                tail_activation_tile(i);
            }
#endif
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        // Release each output row the moment it is complete, exactly as the per-tile form did.
        while (pushed + N_block_tiles <= t + n) {
            cb_push_back(out_cb, N_block_tiles);
            pushed += N_block_tiles;
        }
    }
}

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

    // TRIMUL_TAIL: the two bf16 GEMM results and the rounded gate, one output block each.
    constexpr uint32_t p_cb = tt::CBIndex::c_4;
    constexpr uint32_t g_cb = tt::CBIndex::c_5;

    silu_tile_init();

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

            for (uint32_t pass = 0; pass < TRIMUL_TAIL_PASSES; pass++) {
            const uint32_t pass_cb = (pass == 0) ? p_cb : g_cb;
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

                // in0 reuse across the N stride, corrected for a MULTI-PASS loop.
                //
                // The sender skips exactly one in0 read per strided N block, at `pass == 0 &&
                // k_block_iter == 0` (dm_in0_sender.cpp:265), and pushes one block for every other
                // (n_block, pass, k_block). The wheel's single-pass compute matches that by
                // retaining its block whenever it is about to stride on N. This kernel runs two
                // passes, and the unmodified condition fires at the end of BOTH of them, so pass 1
                // re-reads pass 0's block and every later block on that core is one push behind.
                //
                // MEASURED, before and after: PCC 0.8160 with the condition unmodified as soon as
                // a core folds two output blocks, 0.9999989 at one block a core where the branch
                // cannot fire, and 0.9999989 everywhere once the condition also requires the last
                // pass. Removing the reuse instead of correcting it DEADLOCKS -- the sender is
                // still skipping its read, so the pops stop matching the pushes.
                //
                // What the retained block holds at the strided N block is PASS 1's activation, and
                // pass 0 of the next N block consumes it. That is only sound because both passes
                // of this kernel read the SAME activation tensor (`x_norm` for both projections).
                // `kernels/trimul_tail/compute.cpp` has the unmodified condition and two DIFFERENT
                // activations, so it is not merely one push behind above one block a core, it is
                // also reading the wrong operand. It ships on by default and should be re-validated
                // there.
                if (k_block == K_num_blocks - 1 && pass == TRIMUL_TAIL_PASSES - 1) {
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

            // The fp32 accumulator -> bf16, through the wheel's own copy_block: this is the
            // identical pack that writes p_out and g_out to DRAM in production.
            cb_reserve_back(pass_cb, out_block_num_tiles);
            cb_wait_front(intermediate_cb, out_block_num_tiles);
            copy_block(intermediate_cb, pass_cb, M_block_tiles, N_block_tiles, pass == 1);
            cb_pop_front(intermediate_cb, out_block_num_tiles);
            }  // pass

            cb_reserve_back(out_cb, out_block_num_tiles);
            cb_wait_front(p_cb, out_block_num_tiles);
#if TRIMUL_TAIL_PASSES > 1
            cb_wait_front(g_cb, out_block_num_tiles);
#endif
            // At TRIMUL_TAIL_PASSES == 1 there is no second projection to multiply by, so this is a
            // DIAGNOSTIC build and only MUL_MODE 2 (pack pass 0, no product) is meaningful. Both
            // dataflow kernels honour the same define, so a one-pass build pushes one pass and pops
            // one pass -- nothing waits on a CB that is never filled. It exists to separate the
            // cost of the two-pass LOOP from the cost of the second pass.
            mul_block(p_cb, g_cb, out_cb, out_block_num_tiles);
            cb_pop_front(p_cb, out_block_num_tiles);
#if TRIMUL_TAIL_PASSES > 1
            cb_pop_front(g_cb, out_block_num_tiles);
#endif
        }
    }
}
