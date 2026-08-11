// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute GATED compute: per tile pair, out = transpose_wh(p * sigmoid(g)).
//
// The three phases mirror the two ops this kernel replaces, in the same order and with the same
// rounding points. That is what makes the result bit-exact rather than merely close, and two of the
// choices below are load-bearing for it. Both were measured, not read (perf/trimul_f2/e6_diag.py,
// qb2 card 0, N=320 C=32, forensics over 3.28 M elements):
//
//   * THE DESCRIPTOR MUST RUN 16-BIT DST (`fp32_dest_acc_en=False`). `calculate_sigmoid` branches
//     on that flag: under it, `_sfpu_exp_21f_bf16_` + one reciprocal iteration + an explicit
//     `float_to_fp16b`; over it, `_sfpu_exp_accurate_` + two iterations. ttnn takes the cheap
//     branch, so its sigmoid is a full bf16 ulp off the correctly rounded one on 10.4 % of
//     elements. A kernel compiled with fp32 DST computes the ACCURATE sigmoid and therefore misses.
//   * THE MULTIPLY MUST BE THE SFPU ONE. The FPU's `mul_tiles` truncates the product into a 16-bit
//     DST: 25.7 % of elements land one ulp low, on top of the ties. `mul_binary_tile` rounds, and
//     matches ttnn on every element.
//
// The two knobs interact, which is why the config could not be found by sweeping either alone.
// Packing the product from an fp32 DST instead is the third combination and it is also wrong, in a
// way worth naming because it looks harmless: the packer breaks ties AWAY FROM ZERO where ttnn
// breaks them to even, so exactly half the ties come out one ulp off and nothing else does. That
// is 0.91 % of elements at a 1.85 % tie rate, small enough to read as noise and it is not noise.
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
    // Diagnostic: drop the activation so the multiply can be compared on its own against
    // ttnn.multiply(p, g) with two generic bf16 operands. Never set in production.
    constexpr uint32_t skip_sigmoid = get_compile_time_arg_val(5);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;

    binary_op_init_common(p_cb, sig_cb, mul_cb);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        // sigmoid(g) -> its own CB. binary_ng applies an input activation in PREPROCESS
        // (eltwise_utils.hpp), which copies the operand to DST, runs the SFPU op and packs the
        // result before the binary op unpacks it again, so the activation is rounded to bf16
        // BEFORE the multiply. Keeping it in DST would multiply against more mantissa than ttnn.
        cb_wait_front(g_cb, onetile);
        cb_reserve_back(sig_cb, onetile);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(g_cb);
        copy_tile(g_cb, 0, 0);
        if constexpr (!skip_sigmoid) {
            sigmoid_tile_init();
            sigmoid_tile(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, sig_cb);
        tile_regs_release();
        cb_pop_front(g_cb, onetile);
        cb_push_back(sig_cb, onetile);

        cb_wait_front(p_cb, onetile);
        cb_wait_front(sig_cb, onetile);
        cb_reserve_back(mul_cb, onetile);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(p_cb);
        copy_tile(p_cb, 0, 0);
        copy_tile_to_dst_init_short(sig_cb);
        copy_tile(sig_cb, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, mul_cb);
        tile_regs_release();
        cb_pop_front(p_cb, onetile);
        cb_pop_front(sig_cb, onetile);
        cb_push_back(mul_cb, onetile);

        // The within-tile WH transpose, unchanged from compute_reblock_permute.cpp. It runs last,
        // as it does in production where `_transform_chunk` moves the already-gated chunk.
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
