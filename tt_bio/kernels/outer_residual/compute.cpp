// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// outer_residual compute: RFD4's Hadamard pair write and its residual add as one pass,
//   z'[b, i, j, :] = z[b, i, j, :] + pj[b, i, :] * pi[b, j, :],
// written as float32 (c_16) and as its bfloat16 copy rounded to even (c_17), the copy the next sublayer reads.
// Work items run in (b, jt, i, ct) order (reader.cpp): c_1 holds the current (b, jt) run's KT pi tiles, c_2 one pj4
// row per item. Per tile: z (float32, unpacked straight to DEST) into slot 0; pi's tile times pj4's row 0 broadcast down the rows on
// the FPU into slot 1 (a bf16 x bf16 product is exact at HiFi4 into a 32-bit DEST); the SFPU sum into slot 2, and the
// same sum into slot 0 rounded to even in place (rne_add's DUAL arm, kernels/rne_add/compute_rne_add.cpp). The
// update is never rounded to bfloat16 before the add.
// OR_BF16 (RFD4's bfloat16 pair stream, `pair_bf16z`): z is bfloat16, the rounded sum alone goes to c_16 (bfloat16)
// and there is no c_17.
// OR_BFP8 (with OR_BF16, `pair_bfp8z`): z and c_16 are bfloat8_b and the packer rounds the float32 sum into its 16-value
// blocks directly; no bfloat16 rounding first.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// float32 DEST tile -> bfloat16 precision, ties to even, in place (rne_add's rne_bf16_in_place).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void orz_rne_bf16_in_place() {
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

static_assert(DST_ACCUM_MODE == 1, "outer_residual needs a 32-bit DEST (fp32_dest_acc_en)");

void kernel_main() {
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t RUN = get_compile_time_arg_val(1);   // work items of one (b, jt): N * KT
    constexpr uint32_t KT = get_compile_time_arg_val(2);
    constexpr uint32_t cb_z = 0, cb_i = 1, cb_j = 2, cb_out = 16, cb_lo = 17;
    const uint32_t p0 = get_arg_val<uint32_t>(0);
    const uint32_t np = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_i, cb_j, cb_out);
    uint32_t have = 0xFFFFFFFF;
    for (uint32_t p = p0; p < p0 + np; p += GRAN) {
        const uint32_t n = p0 + np - p < GRAN ? p0 + np - p : GRAN;
        cb_wait_front(cb_z, GRAN);
        cb_wait_front(cb_j, GRAN);
        cb_reserve_back(cb_out, GRAN);
#ifndef OR_BF16
        cb_reserve_back(cb_lo, GRAN);
#endif
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t bj = (p + k) / RUN;
            if (bj != have) {
                if (have != 0xFFFFFFFF) {
                    cb_pop_front(cb_i, KT);
                }
                cb_wait_front(cb_i, KT);
                have = bj;
            }
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(cb_i, cb_z);
            copy_tile(cb_z, k, 0);
            reconfig_data_format(cb_i, cb_j);
            mul_bcast_rows_init_short(cb_i, cb_j);
            mul_tiles_bcast_rows(cb_i, cb_j, (p + k) % KT, k, 1);
            add_binary_tile_init();
#ifndef OR_BF16
            add_binary_tile(0, 1, 2);
#endif
            add_binary_tile(0, 1, 0);
#ifndef OR_BFP8
            MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::bitwise_and, false>()));
            MATH((_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::orz_rne_bf16_in_place<false>, 0,
                                                             static_cast<int>(VectorMode::RC))));
#endif
            tile_regs_commit();
            tile_regs_wait();
#ifdef OR_BF16
            pack_tile(0, cb_out);
#else
            pack_reconfig_data_format(cb_out);
            pack_tile(2, cb_out);
            pack_reconfig_data_format(cb_lo);
            pack_tile(0, cb_lo);
#endif
            tile_regs_release();
        }
        cb_pop_front(cb_z, GRAN);
        cb_pop_front(cb_j, GRAN);
        cb_push_back(cb_out, GRAN);
#ifndef OR_BF16
        cb_push_back(cb_lo, GRAN);
#endif
    }
    if (have != 0xFFFFFFFF) {
        cb_pop_front(cb_i, KT);
    }
}
