// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU helpers shared by the pair kernels: the gates' sigmoid and the transition's silu as polynomials, and a
// round-to-nearest-even of DST to bf16 ahead of a pack.
#pragma once

#include "../common/gate_sigmoid.hpp"

#if defined(TRISC_MATH) || (defined(TRISC_PACK) && defined(WS_PACK_SILU))
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
// Two polynomial sigmoids, sigmoid(x) = 0.5 + sign(x) * h, h(0) = 0, output inside (0, 1):
// V 1: h a degree-8 polynomial in t = min(|x| / 9, 1), fitted (weighted minimax) to sigmoid(9 t) - 0.5 on [0, 1].
//      Max |error| 3.0e-4 over all x in float32 (bf16's half-ULP is 1-2e-3 above 0.25).
// V 2: h a degree-6 polynomial in a = min(|x|, 7) itself, every coefficient exact in fp16
//      (perf/spd_fastmode/silu_fit.py fits it and emulates this float32 Horner order). Max |error| 7.4e-4, so silu
//      is within 7.4e-4 relative for x > 1 and |x| * 7.4e-4 at most below (0.005 at x = -7).
// metal's sigmoid / silu under fp32 dest run the accurate exp and a two-step reciprocal: the sigmoid was 3.3 of
// gin_moved's 7.3 ms at 736 on WH, the silu ~85 % of pair_swiglu_ws (perf/spd_fastmode/ws_attr.py).
//
// Cost is SFPU instruction issue, so the coefficients stay out of the loop: three sit in the programmable constant
// registers (_sig_init_, run per tile because every metal SFPU init reprograms them), the rest load once per pass
// or per face, and two SFPI rows run per pass so each Horner step's MAD latency is covered by the other row's
// instead of an SFPNOP. That needs every LREG, which an out-of-line SFPI function does not get: compiled out of
// line the loop falls back to SFPMOV / SFPNOP chains (no faster than one row) or GCC's reload ICEs, and the JIT
// outlines a face function it receives as a function pointer once a kernel has several call sites. So the tile
// wrappers hand the LLK an always_inline lambda. Executed SFPU instructions per tile at the JIT's -Os against the
// one-row form with every coefficient a 32-bit immediate (perf/sfpu_count --rev 904d0a98e, WH): V 1 sigmoid
// 1440 -> 720, silu / gate 1536 -> 816; V 2 silu 1056 -> 616. Same operations in the same order on every element,
// so the output is bit-identical to that form.
template <uint32_t V>
inline void _sig_init_() {
    if constexpr (V == 2) {
        sfpi::vConstFloatPrgm0 = 0.007404327392578125f;
        sfpi::vConstFloatPrgm1 = -0.0260009765625f;
        sfpi::vConstFloatPrgm2 = -0.0027103424072265625f;
    } else {
        sfpi::vConstFloatPrgm0 = -135.66062927246094f;
        sfpi::vConstFloatPrgm1 = 89.25188446044922f;
        sfpi::vConstFloatPrgm2 = -29.09324073791504f;
    }
}

// t (V 1) or a (V 2) of row x, as the one-row form computes it; k the 1/9 scale (V 1).
template <uint32_t V>
__attribute__((always_inline)) inline sfpi::vFloat _sig_t_(sfpi::vFloat x, sfpi::vFloat k) {
    sfpi::vFloat t = sfpi::abs(x);
    if constexpr (V != 2) t = t * k;
    sfpi::vFloat lim = V == 2 ? 7.0f : 1.0f;
    sfpi::vec_min_max(t, lim);
    return t;
}

// MUL false: row <- sigmoid(row + GOFF). MUL true: row <- row * sigmoid(row + GOFF), i.e. silu at GOFF 0 and the
// gate multiply otherwise (trimul_gin_moved's FUSE path: neither product is packed and read back). GOFF is a template
// argument because the JIT's LTO compiles this out of line, where a runtime DST offset turns every load of it into an
// instruction built in a GPR and stored to the instruction buffer.
template <uint32_t V, bool MUL, uint32_t GOFF = 0, int ITERATIONS = 8>
__attribute__((always_inline)) inline void _sig_rows_() {
    static_assert(ITERATIONS % 2 == 0);
    // Hoisted per face: the two leading coefficients. The 1/9 scale (V 1) loads per pass: hoisted too it is one
    // LREG more than the loop has once the function is compiled out of line, and GCC's reload ICEs.
    const sfpi::vFloat ch = V == 2 ? 3.552436828613281e-05f : 9.57322883605957f;
    const sfpi::vFloat cl = V == 2 ? -0.0008435249328613281f : -51.5362548828125f;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS / 2; d++) {
        const sfpi::vFloat k = V == 2 ? 0.0f : 0.1111111119389534f;
        sfpi::vFloat t0 = _sig_t_<V>(sfpi::dst_reg[GOFF], k);
        sfpi::vFloat t1 = _sig_t_<V>(sfpi::dst_reg[GOFF + 1], k);
        sfpi::vFloat h0 = t0 * ch + cl;
        sfpi::vFloat h1 = t1 * ch + cl;
        if constexpr (V != 2) {
            sfpi::vFloat c = 114.75630187988281f;
            h0 = h0 * t0 + c;
            h1 = h1 * t1 + c;
        }
        h0 = h0 * t0 + sfpi::vConstFloatPrgm0;
        h1 = h1 * t1 + sfpi::vConstFloatPrgm0;
        h0 = h0 * t0 + sfpi::vConstFloatPrgm1;
        h1 = h1 * t1 + sfpi::vConstFloatPrgm1;
        h0 = h0 * t0 + sfpi::vConstFloatPrgm2;
        h1 = h1 * t1 + sfpi::vConstFloatPrgm2;
        if constexpr (V == 2) {
            sfpi::vFloat c = 0.2529296875f;
            h0 = h0 * t0 + c;
            h1 = h1 * t1 + c;
        } else {
            sfpi::vFloat c = 0.9815341234207153f;
            h0 = h0 * t0 + c;
            h1 = h1 * t1 + c;
            c = 2.2268805503845215f;
            h0 = h0 * t0 + c;
            h1 = h1 * t1 + c;
        }
        h0 = sfpi::setsgn(h0 * t0, sfpi::vFloat(sfpi::dst_reg[GOFF])) + 0.5f;
        h1 = sfpi::setsgn(h1 * t1, sfpi::vFloat(sfpi::dst_reg[GOFF + 1])) + 0.5f;
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
#endif  // TRISC_MATH, or TRISC_PACK under WS_PACK_SILU

// V is the kernels' SIGPOLY compile-time arg: 1 the degree-8 polynomial, 2 the degree-6 one, 3 metal's LUT
// sigmoid (TT_BIO_SIGMOID_FAST, gate_sigmoid.hpp).
template <uint32_t V = 1>
ALWI void sigmoid_poly_tile(uint32_t idst) {
    if constexpr (V == 3) {
        gate_sigmoid_lut_tile(idst);
        return;
    }
    MATH((ckernel::sfpu::_sig_init_<V>()));
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        []() __attribute__((always_inline)) { ckernel::sfpu::_sig_rows_<V, false>(); }, idst, (int)VectorMode::RC)));
}

template <uint32_t V = 1>
ALWI void silu_poly_v_tile(uint32_t idst) {
    MATH((ckernel::sfpu::_sig_init_<V>()));
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        []() __attribute__((always_inline)) { ckernel::sfpu::_sig_rows_<V, true>(); }, idst, (int)VectorMode::RC)));
}
ALWI void silu_poly_tile(uint32_t idst) { silu_poly_v_tile<1>(idst); }
ALWI void silu_poly6_tile(uint32_t idst) { silu_poly_v_tile<2>(idst); }

// silu_poly_v_tile issued by the packer thread, on the dest half the packer holds; the caller points the SFPU at that
// half first (pair_swiglu_ws's WS_PACK_SILU, as kernels/fc12g). Same instructions on the same values: bit-identical.
#ifdef WS_PACK_SILU
template <uint32_t V = 2>
ALWI void silu_poly_v_tile_pack(uint32_t idst) {
    PACK((ckernel::sfpu::_sig_init_<V>()));
    PACK((_llk_math_eltwise_unary_sfpu_params_<false>(
        []() __attribute__((always_inline)) { ckernel::sfpu::_sig_rows_<V, true>(); }, idst, (int)VectorMode::RC)));
}
#endif

ALWI void round_bf16_rne_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_round_bf16_rne_<8>, idst, (int)VectorMode::RC)));
}

// DST tile ip <- tile ip * sigmoid(tile ip + GAP), sigmoid as sigmoid_poly_tile<V> (V 1 or 2).
template <uint32_t V, uint32_t GAP>
ALWI void gate_poly_tile(uint32_t ip) {
    MATH((ckernel::sfpu::_sig_init_<V>()));
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        []() __attribute__((always_inline)) { ckernel::sfpu::_sig_rows_<V, true, GAP * 32>(); }, ip, (int)VectorMode::RC)));
}
