// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RFD4's Hadamard pair write formed inside a pair kernel (`pair_outer_fold`, T1OUTER and T2OUTER): one tile of
//   z'[b, i, j, :] = z[b, i, j, :] + pj[b, i, :] * pi[b, j, :]
// into DEST slot 0, rounded to even to bfloat16, by outer_residual's OR_BF16 sequence (kernels/outer_residual): z
// copied to DEST, pi's tile times row 0 of pj4's tile broadcast down the rows on the FPU into slot 1, the SFPU sum
// into slot 0, rounded on its bits. Two details keep it bit-identical to outer_residual: the product runs at HiFi4
// whatever the host kernel's fidelity (bf16 x bf16 is exact into a 32-bit DEST only with all four phases, and T1
// runs at HiFi2 under trimul_gin_hifi2), and the rounding is outer_residual's integer one, not float_to_fp16b,
// which rounds some sums the other way. Needs fp32 DEST. The caller holds the tile registers and packs slot 0.
#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void outer_rne_bf16_in_place() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt v = sfpi::dst_reg[0];
        sfpi::vInt lsb = (v >> 16) & 1;
        v = v + 0x7fff + lsb;
        sfpi::dst_reg[0] = v & static_cast<int>(0xFFFF0000);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
#endif

ALWI void outer_add_tile(uint32_t cb_z, uint32_t tz, uint32_t cb_pi, uint32_t tpi, uint32_t cb_pj, uint32_t tpj) {
    reconfig_data_format_srca(cb_z);
    copy_tile_to_dst_init_short(cb_z);
    copy_tile(cb_z, tz, 0);
    reconfig_data_format(cb_pi, cb_pj);
    MATH((llk_math_eltwise_binary_init_with_operands<ELWMUL, BroadcastType::ROW, MathFidelity::HiFi4>(cb_pi, cb_pj)));
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(cb_pi, cb_pj)));
    MATH((llk_math_eltwise_binary<ELWMUL, BroadcastType::ROW, DST_ACCUM_MODE, MathFidelity::HiFi4,
                                  EltwiseBinaryReuseDestType::NONE>(cb_pi, cb_pj, 1, true)));
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(cb_pi, cb_pj, tpi, tpj, 0)));
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::bitwise_and, false>()));
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::outer_rne_bf16_in_place<false>, 0,
                                                     static_cast<int>(VectorMode::RC))));
}
