#pragma once
#include "ckernel.h"
#include "sfpi.h"
namespace ckernel::sfpu {
inline void silu_f32_init() {
    sfpi::vConstFloatPrgm0 = -1.4426950408889634f;
    sfpi::vConstFloatPrgm1 = 0.693123937f;
    sfpi::vConstFloatPrgm2 = 0.240241066f;
}
// two rows in lockstep so each MAD's 2-cycle latency is covered by the other row's instruction
template <int NEWTON = 2>
inline void calculate_silu_f32_il() {
#pragma GCC unroll 4
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat r0 = sfpi::setsgn(sfpi::vFloat(sfpi::dst_reg[0]), 0) * sfpi::vConstFloatPrgm0;
        sfpi::vFloat r1 = sfpi::setsgn(sfpi::vFloat(sfpi::dst_reg[1]), 0) * sfpi::vConstFloatPrgm0;
        sfpi::vFloat lo0 = -126.0f, lo1 = -126.0f;
        sfpi::vec_min_max(lo0, r0);
        sfpi::vec_min_max(lo1, r1);
        const sfpi::vFloat c231 = 12582912.0f;
        sfpi::vFloat t0 = r0 + c231;
        sfpi::vFloat t1 = r1 + c231;
        r0 = r0 - (t0 - c231);
        r1 = r1 - (t1 - c231);
        sfpi::vInt k0 = (sfpi::reinterpret<sfpi::vInt>(t0) - sfpi::reinterpret<sfpi::vInt>(c231)) << 23;
        sfpi::vInt k1 = (sfpi::reinterpret<sfpi::vInt>(t1) - sfpi::reinterpret<sfpi::vInt>(c231)) << 23;
        sfpi::vFloat p0 = 0.00958251953f * r0 + 0.0559082031f;
        sfpi::vFloat p1 = 0.00958251953f * r1 + 0.0559082031f;
        p0 = p0 * r0 + sfpi::vConstFloatPrgm2;
        p1 = p1 * r1 + sfpi::vConstFloatPrgm2;
        p0 = p0 * r0 + sfpi::vConstFloatPrgm1;
        p1 = p1 * r1 + sfpi::vConstFloatPrgm1;
        p0 = p0 * r0 + sfpi::vConst1;
        p1 = p1 * r1 + sfpi::vConst1;
        sfpi::vFloat e0 = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(p0) + k0);
        sfpi::vFloat e1 = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(p1) + k1);
        sfpi::vFloat y0 = 0.322265625f * e0 + -0.80859375f;
        sfpi::vFloat y1 = 0.322265625f * e1 + -0.80859375f;
        y0 = y0 * e0 + 0.98828125f;
        y1 = y1 * e1 + 0.98828125f;
        sfpi::vFloat n0 = e0 * sfpi::vConstNeg1 + sfpi::vConstNeg1;
        sfpi::vFloat n1 = e1 * sfpi::vConstNeg1 + sfpi::vConstNeg1;
        for (int i = 0; i < NEWTON; i++) {
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
        sfpi::dst_reg[0] = x0 * y0;
        sfpi::dst_reg[1] = x1 * y1;
        sfpi::dst_reg += 2;
    }
}
}
