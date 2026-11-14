// gelu on the SFPU at 3e-4 of float64 (max abs), against 4e-3 for ttnn 0.68's accurate gelu, in about a
// third of its instructions. The wheel's accurate path is a degree-15 Horner polynomial in x with fp32
// coefficients (two SFPLOADI each) under three nested predicates, and it returns x itself from x = 3 on.
//
//   gelu(x) = x/2 + x/2 erf(x/sqrt 2), erf(x/sqrt 2) = sign(x) a P(a^2), a = min(|x|, 4),
//   P of degree 7: least squares weighted by the gelu error, the two top coefficients bfloat16-exact,
//   c0 moved so 4 P(16) = 1 (x/2 + x/2 = x and -x/2 + x/2 = 0 exactly beyond |x| = 4).
// Float64 reference over every bfloat16 with |x| < 8: max abs error 3.0e-4 (at -3.9); rel_l2 on N(0, 1.5)
// 5.8e-5 (the wheel's accurate gelu 3.0e-4). The bfloat16 output tolerance is 2e-3.
//
// Two rows run in lockstep so each MAD's two-cycle latency is covered by the other row. The three
// programmable constants hold c5, c4, c3, so gelu_f32_init() must run before the first call.
#pragma once

#include "ckernel.h"
#include "sfpi.h"

namespace ckernel::sfpu {

inline void gelu_f32_init() {
    sfpi::vConstFloatPrgm0 = -7.5152252065890934e-06f;  // c5
    sfpi::vConstFloatPrgm1 = 0.00015210959827527404f;   // c4
    sfpi::vConstFloatPrgm2 = -0.002028443617746234f;    // c3
}

// a * b + c as one SFPMAD: sfpi's optimizer turns a MAD with a constant addend into MUL, NOP, ADDI.
sfpi_inline sfpi::vFloat gelu_f32_mad(sfpi::vFloat a, sfpi::vFloat b, sfpi::vFloat c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

template <int ITERATIONS>
inline void calculate_gelu_f32() {
    static_assert(ITERATIONS % 2 == 0, "calculate_gelu_f32 runs two rows per step");
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat a0 = sfpi::setsgn(sfpi::vFloat(sfpi::dst_reg[0]), 0);
        sfpi::vFloat a1 = sfpi::setsgn(sfpi::vFloat(sfpi::dst_reg[1]), 0);
        sfpi::vFloat hi0 = 4.0f, hi1 = 4.0f;
        sfpi::vec_min_max(a0, hi0);  // a = min(|x|, 4)
        sfpi::vec_min_max(a1, hi1);
        sfpi::vFloat t0 = a0 * a0;
        sfpi::vFloat t1 = a1 * a1;
        sfpi::vFloat c7 = -2.7503119781613350e-09f, c6 = 2.1699815988540650e-07f;  // bfloat16-exact
        sfpi::vFloat p0 = gelu_f32_mad(c7, t0, c6);
        sfpi::vFloat p1 = gelu_f32_mad(c7, t1, c6);
        p0 = p0 * t0 + sfpi::vConstFloatPrgm0;
        p1 = p1 * t1 + sfpi::vConstFloatPrgm0;
        p0 = p0 * t0 + sfpi::vConstFloatPrgm1;
        p1 = p1 * t1 + sfpi::vConstFloatPrgm1;
        p0 = p0 * t0 + sfpi::vConstFloatPrgm2;
        p1 = p1 * t1 + sfpi::vConstFloatPrgm2;
        sfpi::vFloat c2 = 0.019062716513872147f;
        p0 = gelu_f32_mad(p0, t0, c2);
        p1 = gelu_f32_mad(p1, t1, c2);
        sfpi::vFloat c1 = -0.13185493648052216f;
        p0 = gelu_f32_mad(p0, t0, c1);
        p1 = gelu_f32_mad(p1, t1, c1);
        sfpi::vFloat c0 = 0.7974156737327576f;
        p0 = gelu_f32_mad(p0, t0, c0);
        p1 = gelu_f32_mad(p1, t1, c0);
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        sfpi::vFloat e0 = sfpi::setsgn(a0 * p0, x0);  // erf(x / sqrt 2)
        sfpi::vFloat e1 = sfpi::setsgn(a1 * p1, x1);
        sfpi::vFloat h0 = x0 * 0.5f;
        sfpi::vFloat h1 = x1 * 0.5f;
        sfpi::dst_reg[0] = h0 * e0 + h0;
        sfpi::dst_reg[1] = h1 * e1 + h1;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
