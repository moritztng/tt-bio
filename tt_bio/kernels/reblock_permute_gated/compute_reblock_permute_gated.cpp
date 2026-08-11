// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute GATED compute: per tile pair, out = transpose_wh(p * sigmoid(g)).
//
// The three phases mirror what the two ops this kernel replaces do, in the same order and with
// the same rounding points, which is what makes the result bit-exact rather than merely close:
//
//   1. sigmoid(g) into DST, PACKED BACK TO A CB. binary_ng applies an input activation in
//      PREPROCESS (eltwise_utils.hpp), which copies the operand to DST, runs the SFPU op and packs
//      the result into cb_post_rhs before the binary op unpacks it again. The activation result is
//      therefore rounded to the CB's format before the multiply, and a version that kept it in DST
//      would multiply against more mantissa than ttnn does.
//   2. mul_tiles(p, sigmoid(g)) -> DST -> CB. Same FPU multiply ttnn issues.
//
// The two phases want OPPOSITE dest accumulate modes and that is the whole parity story here.
// Measured on qb2 card 0 at N=320, C=32 (perf/trimul_f2/e6_diag.py):
//
//   * the MULTIPLY needs fp32 DST. In 16-bit DST the exact bf16 x bf16 product is truncated on the
//     way in, and 26.6 % of elements land one ulp below `ttnn.multiply_`, which is itself exactly
//     the correctly rounded product (R0_vs_m_exact: equal).
//   * the SIGMOID needs the 16-bit algorithm. `calculate_sigmoid` branches on the same flag and
//     picks `_sfpu_exp_21f_bf16_` + one reciprocal iteration + an explicit `float_to_fp16b` under
//     it, against `_sfpu_exp_accurate_` + two iterations above it. ttnn takes the cheap branch:
//     its sigmoid is a full bf16 ulp off the correctly rounded one on 10.4 % of elements, and a
//     kernel compiled with fp32 DST reproduces the ACCURATE value and so misses ttnn's.
//
// One kernel has one dest mode, but the flag is only a template argument. So the kernel is
// compiled with fp32 DST for the multiply and the sigmoid is called through the LLK with the flag
// forced false, which runs the 16-bit algorithm and rounds its own result to bf16 in software.
// Storing that already-rounded value in an fp32 DST is exact, so the two phases each get what they
// need. `sigmoid_tile()` cannot express this: it hardwires the template argument to the kernel's
// own DST_ACCUM_MODE.
//   3. the within-tile WH transpose, unchanged from compute_reblock_permute.cpp. It runs LAST, as
//      it does in production, where `_transform_chunk` moves the already-gated chunk. A permute is
//      a pure index reordering, so its position cannot change a value, but keeping the order also
//      keeps the diff against the ungated kernel readable.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    constexpr uint32_t p_cb = get_compile_time_arg_val(0);    // c_0,  value slice
    constexpr uint32_t g_cb = get_compile_time_arg_val(1);    // c_1,  gate slice
    constexpr uint32_t sig_cb = get_compile_time_arg_val(2);  // c_2,  sigmoid(g)
    constexpr uint32_t mul_cb = get_compile_time_arg_val(3);  // c_3,  p * sigmoid(g)
    constexpr uint32_t out_cb = get_compile_time_arg_val(4);  // c_16, post-WH, the writer's input
    // binary_ng ships an FPU kernel and an SFPU kernel for the same op and picks between them
    // host-side, out of reach from here, so which one `ttnn.multiply_` ran is a measurement rather
    // than a reading. Both are compiled in and the parity probe decides.
    constexpr uint32_t sfpu_mul = get_compile_time_arg_val(5);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;

    binary_op_init_common(p_cb, sig_cb, mul_cb);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(g_cb, onetile);
        cb_reserve_back(sig_cb, onetile);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(g_cb);
        copy_tile(g_cb, 0, 0);
        MATH((ckernel::llk_math_eltwise_unary_sfpu_sigmoid_init<false>()));
        MATH((ckernel::llk_math_eltwise_unary_sfpu_sigmoid<false, /*is_fp32_dest_acc_en=*/false>(
            0, (int)ckernel::VectorMode::RC)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, sig_cb);
        tile_regs_release();
        cb_pop_front(g_cb, onetile);
        cb_push_back(sig_cb, onetile);

        cb_wait_front(p_cb, onetile);
        cb_wait_front(sig_cb, onetile);
        cb_reserve_back(mul_cb, onetile);
        if constexpr (sfpu_mul) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(p_cb);
            copy_tile(p_cb, 0, 0);
            copy_tile_to_dst_init_short(sig_cb);
            copy_tile(sig_cb, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
        } else {
            mul_tiles_init(p_cb, sig_cb);
            tile_regs_acquire();
            mul_tiles(p_cb, sig_cb, 0, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, mul_cb);
        tile_regs_release();
        cb_pop_front(p_cb, onetile);
        cb_pop_front(sig_cb, onetile);
        cb_push_back(mul_cb, onetile);

        cb_wait_front(mul_cb, onetile);
        cb_reserve_back(out_cb, onetile);
        transpose_wh_init(mul_cb, out_cb);
        tile_regs_acquire();
        transpose_wh_tile(mul_cb, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_pop_front(mul_cb, onetile);
        cb_push_back(out_cb, onetile);
    }
}
