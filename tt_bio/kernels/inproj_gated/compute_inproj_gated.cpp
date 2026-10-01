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

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
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
}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_MATH

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
        cb_wait_front(cb_x, TILE_HEIGHT * Kt);
        for (uint32_t q = 0; q < CPG; ++q) {
            const uint32_t ct = sub * CPG + q;
            const uint32_t wp = ct * Kt1;
            const uint32_t wg = (Ct + ct) * Kt1;
            cb_reserve_back(cb_out, TILE_HEIGHT);
            for (uint32_t il = 0; il < TILE_HEIGHT; il += ROWS) {
                tile_regs_acquire();
                mm_init_short(cb_w, cb_x, 1);
                for (uint32_t r = 0; r < ROWS; ++r) {
                    const uint32_t xr = (il + r) * Kt;
                    for (uint32_t k = 0; k < Kt; ++k) {
                        matmul_tiles(cb_w, cb_x, wp + k, xr + k, 2 * r);
                    }
                    for (uint32_t k = 0; k < Kt; ++k) {
                        matmul_tiles(cb_w, cb_x, wg + k, xr + k, 2 * r + 1);
                    }
                    if constexpr (HAS_BIAS) {
                        matmul_tiles(cb_w, cb_ones, wp + Kt, 0, 2 * r);
                        matmul_tiles(cb_w, cb_ones, wg + Kt, 0, 2 * r + 1);
                    }
                }
                sigmoid_tile_init();
                for (uint32_t r = 0; r < ROWS; ++r) {
                    sigmoid_21f_tile(2 * r + 1);
                }
                mul_binary_tile_init();
                for (uint32_t r = 0; r < ROWS; ++r) {
                    mul_binary_tile(2 * r, 2 * r + 1, 2 * r);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t r = 0; r < ROWS; ++r) {
                    pack_tile(2 * r, cb_out);
                }
                tile_regs_release();
            }
            cb_push_back(cb_out, TILE_HEIGHT);
        }
        cb_pop_front(cb_x, TILE_HEIGHT * Kt);

        group += group_stride;
        if (group == group_wrap_hi) {
            group = group_wrap_lo;
        }
    }
}
