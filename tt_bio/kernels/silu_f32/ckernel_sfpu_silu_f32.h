// silu on the SFPU at a few ulp of float32 in 32 instructions per row, against 92 for ttnn 0.68's
// fp32-dest silu (Cody-Waite exp + two Newton reciprocals) and 49 for its bfloat16 path.
//
//   e = exp(-|x|) = 2^z, z = -|x| log2(e) >= -126, by z = k + r (k = round(z), |r| <= 1/2) and a
//       degree-4 minimax 2^r (relative error 2.9e-6, the two top coefficients bfloat16-exact),
//       put together as an integer add of k into the exponent field (k << 23 is bits(z + 1.5 2^23) << 23:
//       the constant's low nine bits are zero);
//   s = 1/(1+e) on [1/2, 1) from a bfloat16 quadratic seed in e and two Newton steps (4.5e-8);
//   silu = x s for x >= 0, x e s for x < 0 (sigmoid(x) = e/(1+e) there, no cancellation).
// Float64 reference over every bfloat16 in [-90, 90]: max relative error 6.5e-6, 3.6e-6 for N(0,3)
// inputs (perf/spd_swiglu/silu_emu.py). The bfloat16 output tolerance is 2e-3.
//
// Two rows run in lockstep so each MAD's two-cycle latency is covered by the other row; x is
// reloaded from dest at the end instead of held, which keeps both rows inside the eight LREGs. The
// constant-addend MADs go through mad(): left to itself sfpi emits MUL, NOP, ADDI for each. What is
// left of the NOPs is the one SFPSWAP needs after it.
// The three programmable constants hold -log2(e) and the two fp32 coefficients of 2^r, so
// silu_f32_init() must run before the first call (it replaces the reciprocal's constants).
#pragma once

#include "ckernel.h"
#include "sfpi.h"

namespace ckernel::sfpu {

inline void silu_f32_init() {
    sfpi::vConstFloatPrgm0 = -1.4426950408889634f;
    sfpi::vConstFloatPrgm1 = 0.693123937f;
    sfpi::vConstFloatPrgm2 = 0.240241066f;
}

// a * b + c as one SFPMAD: sfpi's optimizer turns a MAD with a constant addend into MUL, NOP, ADDI.
sfpi_inline sfpi::vFloat mad(sfpi::vFloat a, sfpi::vFloat b, sfpi::vFloat c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu_f32() {
    static_assert(ITERATIONS % 2 == 0, "calculate_silu_f32 runs two rows per step");
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat z0 = sfpi::setsgn(sfpi::vFloat(sfpi::dst_reg[0]), 0) * sfpi::vConstFloatPrgm0;
        sfpi::vFloat z1 = sfpi::setsgn(sfpi::vFloat(sfpi::dst_reg[1]), 0) * sfpi::vConstFloatPrgm0;
        sfpi::vFloat lo0 = -126.0f, lo1 = -126.0f;
        sfpi::vec_min_max(lo0, z0);
        sfpi::vec_min_max(lo1, z1);
        const sfpi::vFloat c231 = 12582912.0f;
        sfpi::vFloat t0 = z0 + c231;
        sfpi::vFloat t1 = z1 + c231;
        sfpi::vFloat q0 = mad(t0, sfpi::vConstNeg1, c231);  // -round(z)
        sfpi::vFloat q1 = mad(t1, sfpi::vConstNeg1, c231);
        sfpi::vFloat r0 = z0 + q0;
        sfpi::vFloat r1 = z1 + q1;
        // bits(c231) has nine zero low bits, so bits(t) << 23 is k << 23
        sfpi::vInt k0 = sfpi::reinterpret<sfpi::vInt>(t0) << 23;
        sfpi::vInt k1 = sfpi::reinterpret<sfpi::vInt>(t1) << 23;
        sfpi::vFloat c4 = 0.00958251953f, c3 = 0.0559082031f;
        sfpi::vFloat p0 = mad(c4, r0, c3);
        sfpi::vFloat p1 = mad(c4, r1, c3);
        p0 = p0 * r0 + sfpi::vConstFloatPrgm2;
        p1 = p1 * r1 + sfpi::vConstFloatPrgm2;
        p0 = p0 * r0 + sfpi::vConstFloatPrgm1;
        p1 = p1 * r1 + sfpi::vConstFloatPrgm1;
        p0 = p0 * r0 + sfpi::vConst1;
        p1 = p1 * r1 + sfpi::vConst1;
        sfpi::vFloat e0 = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(p0) + k0);
        sfpi::vFloat e1 = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(p1) + k1);
        sfpi::vFloat s2 = 0.322265625f, s1 = -0.80859375f;
        sfpi::vFloat y0 = mad(s2, e0, s1);
        sfpi::vFloat y1 = mad(s2, e1, s1);
        sfpi::vFloat s0 = 0.98828125f;
        y0 = mad(y0, e0, s0);
        y1 = mad(y1, e1, s0);
        sfpi::vFloat n0 = e0 * sfpi::vConstNeg1 + sfpi::vConstNeg1;
        sfpi::vFloat n1 = e1 * sfpi::vConstNeg1 + sfpi::vConstNeg1;
#pragma GCC unroll 2
        for (int i = 0; i < 2; i++) {
            sfpi::vFloat u0 = n0 * y0 + sfpi::vConst1;
            sfpi::vFloat u1 = n1 * y1 + sfpi::vConst1;
            y0 = y0 * u0 + y0;
            y1 = y1 * u1 + y1;
        }
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        v_if (x0 < 0.0f) { y0 = y0 * e0; }
        v_endif;
        v_if (x1 < 0.0f) { y1 = y1 * e1; }
        v_endif;
        sfpi::vFloat o0 = x0 * y0;
        sfpi::vFloat o1 = x1 * y1;
        if constexpr (!is_fp32_dest_acc_en) {
            o0 = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(o0, 0));
            o1 = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(o1, 0));
        }
        sfpi::dst_reg[0] = o0;
        sfpi::dst_reg[1] = o1;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
