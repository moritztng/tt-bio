// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU helpers shared by trimul_gin_moved and trimul_tail_res: the gates' sigmoid as a polynomial, and a
// round-to-nearest-even of DST to bf16 ahead of a pack.
#pragma once

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
// sigmoid(x) = 0.5 + sign(x) * h(t), t = min(|x| / 9, 1), h a degree-8 polynomial with h(0) = 0 fitted
// (weighted minimax) to sigmoid(9 t) - 0.5 on [0, 1]. Max |error| 3.0e-4 over all x in float32 (bf16's
// half-ULP is 1-2e-3 above 0.25), output inside (0, 1). ~25 SFPU instructions against ~55 for
// exp_21f + reciprocal, and the sigmoid was 3.3 of gin_moved's 7.3 ms at 736 on WH (stage ablation).
template <int ITERATIONS = 8>
inline void _sigmoid_poly_() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat t = sfpi::abs(x) * 0.1111111119389534f;
        v_if(t > 1.0f) { t = 1.0f; }
        v_endif;
        // Coefficients as immediates: hoisting all eight into LREGs leaves too few for the loop
        // (SFPI's reload pass gives up).
        sfpi::vFloat h = t * 9.57322883605957f - 51.5362548828125f;
        h = h * t + 114.75630187988281f;
        h = h * t - 135.66062927246094f;
        h = h * t + 89.25188446044922f;
        h = h * t - 29.09324073791504f;
        h = h * t + 0.9815341234207153f;
        h = h * t + 2.2268805503845215f;
        h = h * t;
        sfpi::dst_reg[0] = sfpi::setsgn(h, x) + 0.5f;
        sfpi::dst_reg++;
    }
}

// Round DST to bf16, nearest-even, so the pack that follows is exact. The fp32 -> bf16 pack from DST
// truncates: the resident tail's z + p * sigmoid(g) came out a full bf16 ULP short toward zero on
// 0.18 % of elements, rel_rms 0.0024 to float64 where one rounding gives 0.0014 (epi2_diag.py).
template <int ITERATIONS = 8>
inline void _round_bf16_rne_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(v, sfpi::RoundMode::NearestEven));
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_MATH

ALWI void sigmoid_poly_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_sigmoid_poly_<8>, idst, (int)VectorMode::RC)));
}

ALWI void round_bf16_rne_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_round_bf16_rne_<8>, idst, (int)VectorMode::RC)));
}
