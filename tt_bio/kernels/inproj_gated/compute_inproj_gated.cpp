// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// inproj_gated compute: per (position row i, channel tile ct) of one role,
//   out = (W_p^T @ x_i^T + b_p) * sigmoid(W_g^T @ x_i^T + b_g),
// a c x j tile, which is the tile the gated move's transpose_wh produced, so the move's writer
// takes it unchanged. matmul_tiles with in1 transposed gives W^T (c x k) @ x^T (k x j) directly.
//
// Everything stays in a float32 DST between the matmul and the single bf16 pack: the two-op path
// rounds the projection to bf16 in DRAM and the sigmoid to bf16 in a CB first. Not bit-exact
// against it, and graded against float64 instead. The sigmoid is the LLK's own formula,
// 1 / (1 + exp(-g)) with the bf16-accurate exp_21f and one reciprocal Newton step, which is the
// branch a 16-bit DST takes. The float32 DST would otherwise select the accurate exp and two
// Newton steps, ~4x the SFPU time, for bits the bf16 pack discards.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_sigmoid.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
template <int ITERATIONS = 8>
inline void _inproj_sigmoid_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _sfpu_sigmoid_<false>(v);
        sfpi::dst_reg++;
    }
}
// The whole gate in one SFPU pass over a (p, g) tile pair: p * 1 / (1 + exp(-g)), in place on p.
// One load of each operand and one store, where sigmoid then mul_binary_tile is two passes and two
// inits per DST half. GATE_RECIP_ITERS Newton steps after the hardware reciprocal estimate.
#ifndef GATE_RECIP_ITERS
#define GATE_RECIP_ITERS 1
#endif
template <int ITERATIONS = 8>
inline void _inproj_gate_(const uint dst_p, const uint dst_g, const uint /*dst_out*/) {
    constexpr uint dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[dst_g * dst_tile_size_sfpi];
        sfpi::vFloat p = sfpi::dst_reg[dst_p * dst_tile_size_sfpi];
        sfpi::vFloat e = _sfpu_exp_21f_bf16_<true>(-g);
        sfpi::vFloat s = _sfpu_reciprocal_<GATE_RECIP_ITERS>(sfpi::vConst1 + e);
        sfpi::dst_reg[dst_p * dst_tile_size_sfpi] = p * s;
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_MATH

ALWI void gate_tile(uint32_t dst_p, uint32_t dst_g) {
    MATH((_llk_math_eltwise_binary_sfpu_params_<false>(
        ckernel::sfpu::_inproj_gate_<8>, dst_p, dst_g, dst_p, (int)VectorMode::RC)));
}

ALWI void sigmoid_21f_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_inproj_sigmoid_<8>, idst, (int)VectorMode::RC)));
}

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t Kt = get_compile_time_arg_val(4);
    constexpr uint32_t Ct = get_compile_time_arg_val(5);
    constexpr uint32_t HAS_BIAS = get_compile_time_arg_val(6);
    constexpr uint32_t cb_slab = get_compile_time_arg_val(7);
    constexpr uint32_t Kt1 = Kt + HAS_BIAS;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t ROWS = 2;  // two rows a DST half: p and g each, four float32 tiles

    const uint32_t first_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t S = get_arg_val<uint32_t>(2);
    const uint32_t group_stride = get_arg_val<uint32_t>(3);
    const uint32_t group_wrap_hi = get_arg_val<uint32_t>(4);
    const uint32_t group_wrap_lo = get_arg_val<uint32_t>(5);
    const uint32_t CPG = Ct / S;

    mm_init(cb_w, cb_x, cb_out, 1);
    cb_wait_front(cb_w, 2 * Ct * Kt1);
    if constexpr (HAS_BIAS) {
        cb_wait_front(cb_ones, 1);
    }

    uint32_t group = first_group;
    for (uint32_t gi = 0; gi < num_groups; ++gi) {
        const uint32_t sub = group % S;
        // The first channel tile waits row by row as the reader lands them (cumulative waits on
        // the same front); the rest find the whole group resident.
        for (uint32_t q = 0; q < CPG; ++q) {
            const uint32_t ct = sub * CPG + q;
            const uint32_t wp = ct * Kt1;
            const uint32_t wg = (Ct + ct) * Kt1;
#ifdef TILIZE_PATH
            // The reblock on the TRISCs instead of 2048 face-row reads on the writer per channel
            // tile: each c x j tile T_i is packed UNTILIZED as column block i of a 32-tile-wide
            // row-major slab, so slab row c holds [T_0 row c | T_1 row c | ...]. For every c the
            // 32 x 32 block (i, j) is then contiguous in 2 KB, and tilizing those 32 pages one at
            // a time gives the output tiles (c, it, jt) directly. DST: p at 0,1 and g at 2,3, so
            // the two gated rows sit in dst 0,1 as one 2-tile untilize block.
            cb_reserve_back(cb_slab, TILE_HEIGHT);
            pack_untilize_dest_init<ROWS, TILE_HEIGHT>(cb_slab);
#else
            cb_reserve_back(cb_out, TILE_HEIGHT);
#endif
            for (uint32_t il = 0; il < TILE_HEIGHT; il += ROWS) {
                if (q == 0) {
                    cb_wait_front(cb_x, (il + ROWS) * Kt);
                }
                tile_regs_acquire();
                mm_init_short(cb_w, cb_x, 1);
                for (uint32_t r = 0; r < ROWS; ++r) {
                    const uint32_t xr = (il + r) * Kt;
#ifdef DIAG_ONE_K
                    constexpr uint32_t KK = 1;  // diagnostic only: output is wrong
#else
                    constexpr uint32_t KK = Kt;
#endif
                    for (uint32_t k = 0; k < KK; ++k) {
                        matmul_tiles(cb_w, cb_x, wp + k, xr + k, r);
                    }
                    for (uint32_t k = 0; k < KK; ++k) {
                        matmul_tiles(cb_w, cb_x, wg + k, xr + k, ROWS + r);
                    }
                    if constexpr (HAS_BIAS) {
                        matmul_tiles(cb_w, cb_ones, wp + Kt, 0, r);
                        matmul_tiles(cb_w, cb_ones, wg + Kt, 0, ROWS + r);
                    }
                }
#if defined(GATE_FUSED) && !defined(DIAG_NO_SFPU)
                // sigmoid_tile_init only for its reciprocal constants and the SFPU address modes.
                sigmoid_tile_init();
                for (uint32_t r = 0; r < ROWS; ++r) {
                    gate_tile(r, ROWS + r);
                }
#elif !defined(DIAG_NO_SFPU)
                sigmoid_tile_init();
                for (uint32_t r = 0; r < ROWS; ++r) {
                    sigmoid_21f_tile(ROWS + r);
                }
                mul_binary_tile_init();
                for (uint32_t r = 0; r < ROWS; ++r) {
                    mul_binary_tile(r, ROWS + r, r);
                }
#endif
                tile_regs_commit();
                tile_regs_wait();
#ifdef TILIZE_PATH
                pack_untilize_dest<ROWS, TILE_HEIGHT>(cb_slab, 1, il / ROWS);
#else
                for (uint32_t r = 0; r < ROWS; ++r) {
                    pack_tile(r, cb_out);
                }
#endif
                tile_regs_release();
            }
#ifdef TILIZE_PATH
            pack_untilize_uninit(cb_slab);
            cb_push_back(cb_slab, TILE_HEIGHT);
            tilize_init(cb_slab, 1, cb_out);
            cb_wait_front(cb_slab, TILE_HEIGHT);
            cb_reserve_back(cb_out, TILE_HEIGHT);
            for (uint32_t c = 0; c < TILE_HEIGHT; ++c) {
                tilize_block(cb_slab, 1, cb_out, c, c);
            }
            cb_push_back(cb_out, TILE_HEIGHT);
            cb_pop_front(cb_slab, TILE_HEIGHT);
            tilize_uninit(cb_slab, cb_out);
#else
            cb_push_back(cb_out, TILE_HEIGHT);
#endif
        }
        cb_pop_front(cb_x, TILE_HEIGHT * Kt);

        group += group_stride;
        if (group == group_wrap_hi) {
            group = group_wrap_lo;
        }
    }
}
