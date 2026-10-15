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
// half-ULP is 1-2e-3 above 0.25), output inside (0, 1). The sigmoid was 3.3 of gin_moved's 7.3 ms at 736 on WH.
//
// Cost is SFPU instruction issue, so the coefficients stay out of the loop: three sit in the programmable
// constant registers (_sig8_init_, run per tile because every metal SFPU init reprograms them), three are
// hoisted into LREGs per face, and two SFPI rows run per pass so each Horner step's MAD latency is covered by
// the other row's instead of an SFPNOP. 45 instructions per two rows against 46 per row with every coefficient
// loaded as a 32-bit immediate (riscv-tt-elf objdump, WH and BH). Same operations in the same order on every
// element, so the result is bit-identical to the one-row form.
inline void _sig8_init_() {
    sfpi::vConstFloatPrgm0 = -135.66062927246094f;
    sfpi::vConstFloatPrgm1 = 89.25188446044922f;
    sfpi::vConstFloatPrgm2 = -29.09324073791504f;
}

// MUL false: row <- sigmoid(row + goff). MUL true: row <- row * sigmoid(row + goff) (silu at goff 0, the gate
// multiply otherwise).
template <bool MUL, int ITERATIONS = 8>
inline void _sig8_rows_(uint32_t goff) {
    static_assert(ITERATIONS % 2 == 0);
    const sfpi::vFloat k9 = 0.1111111119389534f, c8 = 9.57322883605957f, c7 = -51.5362548828125f;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat t0 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[goff])) * k9;
        sfpi::vFloat t1 = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[goff + 1])) * k9;
        sfpi::vFloat one = 1.0f;
        sfpi::vec_min_max(t0, one);
        one = 1.0f;
        sfpi::vec_min_max(t1, one);
        sfpi::vFloat h0 = t0 * c8 + c7;
        sfpi::vFloat h1 = t1 * c8 + c7;
        sfpi::vFloat c = 114.75630187988281f;
        h0 = h0 * t0 + c;
        h1 = h1 * t1 + c;
        h0 = h0 * t0 + sfpi::vConstFloatPrgm0;
        h1 = h1 * t1 + sfpi::vConstFloatPrgm0;
        h0 = h0 * t0 + sfpi::vConstFloatPrgm1;
        h1 = h1 * t1 + sfpi::vConstFloatPrgm1;
        h0 = h0 * t0 + sfpi::vConstFloatPrgm2;
        h1 = h1 * t1 + sfpi::vConstFloatPrgm2;
        c = 0.9815341234207153f;
        h0 = h0 * t0 + c;
        h1 = h1 * t1 + c;
        c = 2.2268805503845215f;
        h0 = h0 * t0 + c;
        h1 = h1 * t1 + c;
        h0 = sfpi::setsgn(h0 * t0, sfpi::vFloat(sfpi::dst_reg[goff])) + 0.5f;
        h1 = sfpi::setsgn(h1 * t1, sfpi::vFloat(sfpi::dst_reg[goff + 1])) + 0.5f;
        if constexpr (MUL) {
            sfpi::dst_reg[0] = sfpi::dst_reg[0] * h0;
            sfpi::dst_reg[1] = sfpi::dst_reg[1] * h1;
        } else {
            sfpi::dst_reg[0] = h0;
            sfpi::dst_reg[1] = h1;
        }
        sfpi::dst_reg += 2;
    }
}

// sigmoid on _silu_poly6_'s degree-6 fit (lever trimul_sig6): max |error| 7.4e-4 against float64, output inside
// (0, 1), ~2/3 of _sigmoid_poly_'s SFPU instructions (every coefficient one SFPLOADI, no 1/9 scale).
template <int ITERATIONS = 8>
inline void _sigmoid_poly6_() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat a = sfpi::abs(x);
        v_if(a > 7.0f) { a = 7.0f; }
        v_endif;
        sfpi::vFloat h = a * 3.552436828613281e-05f - 0.0008435249328613281f;
        h = h * a + 0.007404327392578125f;
        h = h * a - 0.0260009765625f;
        h = h * a - 0.0027103424072265625f;
        h = h * a + 0.2529296875f;
        h = h * a;
        sfpi::dst_reg[0] = sfpi::setsgn(h, x) + 0.5f;
        sfpi::dst_reg++;
    }
}

// silu(x) = x * sigmoid(x) on the polynomial above is _sig8_rows_<true> at goff 0 (pair_swiglu_ws). Same error
// bound on the sigmoid factor; the metal silu_tile under fp32 dest runs the accurate exp and a two-step reciprocal
// instead, and was ~85 % of that kernel on WH (perf/spd_fastmode/ws_attr.py).

// silu on a cheaper sigmoid: h a degree-6 polynomial in a = min(|x|, 7) itself, with h(0) = 0 and every
// coefficient exact in fp16, so each loads in one SFPLOADI and no 1/X scale is needed (perf/spd_fastmode/silu_fit.py
// fits it and emulates this float32 Horner order). Sigmoid factor within 7.4e-4 over all float32 x, so silu is
// within 7.4e-4 relative for x > 1, under bf16's half-ULP, which the gate is packed to; for negative x the error is
// |x| * 7.4e-4 at most (0.005 at x = -7). ~2/3 of _silu_poly_'s SFPU instructions.
template <int ITERATIONS = 8>
inline void _silu_poly6_() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat a = sfpi::abs(x);
        v_if(a > 7.0f) { a = 7.0f; }
        v_endif;
        sfpi::vFloat h = a * 3.552436828613281e-05f - 0.0008435249328613281f;
        h = h * a + 0.007404327392578125f;
        h = h * a - 0.0260009765625f;
        h = h * a - 0.0027103424072265625f;
        h = h * a + 0.2529296875f;
        h = h * a;
        sfpi::dst_reg[0] = x * (sfpi::setsgn(h, x) + 0.5f);
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
// p * sigmoid(g) in place of p, g `goff` SFPI rows further on in DST (32 per tile), on the degree-6 sigmoid: the
// gate's multiply done in DST, so neither product is packed and read back (trimul_gin_moved's FUSE path). The
// degree-8 form is _sig8_rows_<true>.
template <int ITERATIONS = 8>
inline void _gate_poly6_(uint32_t goff) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[goff];
        sfpi::vFloat a = sfpi::abs(x);
        v_if(a > 7.0f) { a = 7.0f; }
        v_endif;
        sfpi::vFloat h = a * 3.552436828613281e-05f - 0.0008435249328613281f;
        h = h * a + 0.007404327392578125f;
        h = h * a - 0.0260009765625f;
        h = h * a - 0.0027103424072265625f;
        h = h * a + 0.2529296875f;
        h = h * a;
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * (sfpi::setsgn(h, x) + 0.5f);
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_MATH

// V is the kernels' SIGPOLY compile-time arg: 1 the degree-8 polynomial, 2 the degree-6 one.
template <uint32_t V = 1>
ALWI void sigmoid_poly_tile(uint32_t idst) {
    if constexpr (V == 2) {
        MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
            ckernel::sfpu::_sigmoid_poly6_<8>, idst, (int)VectorMode::RC)));
    } else {
        MATH((ckernel::sfpu::_sig8_init_()));
        MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
            ckernel::sfpu::_sig8_rows_<false, 8>, idst, (int)VectorMode::RC, 0u)));
    }
}

ALWI void silu_poly_tile(uint32_t idst) {
    MATH((ckernel::sfpu::_sig8_init_()));
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_sig8_rows_<true, 8>, idst, (int)VectorMode::RC, 0u)));
}

ALWI void silu_poly6_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_silu_poly6_<8>, idst, (int)VectorMode::RC)));
}

ALWI void round_bf16_rne_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_round_bf16_rne_<8>, idst, (int)VectorMode::RC)));
}

// DST tile ip <- tile ip * sigmoid(tile ig), ig > ip, sigmoid as sigmoid_poly_tile<V> (V 1 or 2).
template <uint32_t V = 1>
ALWI void gate_poly_tile(uint32_t ip, uint32_t ig) {
    if constexpr (V == 2) {
        MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
            ckernel::sfpu::_gate_poly6_<8>, ip, (int)VectorMode::RC, (ig - ip) * 32)));
    } else {
        MATH((ckernel::sfpu::_sig8_init_()));
        MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
            ckernel::sfpu::_sig8_rows_<true, 8>, ip, (int)VectorMode::RC, (ig - ip) * 32)));
    }
}
