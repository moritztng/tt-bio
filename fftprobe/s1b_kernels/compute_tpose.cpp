// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Is `transpose_wh_dest` bit-exact in fp32? This is the obstacle that blocked tt-metal #21412 for
// 15 months: "the precision is seriously affected by the lack of full precision fp32 transpose".
// `ttnn.transpose` returns relative L2 4.15e-4 because the tile transpose goes through the FPU,
// whose srcA/srcB registers truncate fp32 to about 11 mantissa bits. `transpose_wh_dest` operates
// in place on the DST register instead, which under fp32_dest_acc_en holds full fp32 -- so it may
// never cross the truncating path at all. Arm 1 is the FPU transpose for a same-harness control.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/transpose_wh_dest.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t is_32bit = get_compile_time_arg_val(2);
    constexpr uint32_t use_fpu = get_compile_time_arg_val(3);   // 1 = transpose_wh_tile, the control

    cb_wait_front(in_cb, 2);
    cb_reserve_back(out_cb, 1);

    if constexpr (use_fpu) {
        transpose_wh_init(in_cb, out_cb);
        tile_regs_acquire();
        transpose_wh_tile(in_cb, 0, 0);
        tile_regs_commit();
    } else {
        binary_op_init_common(in_cb, in_cb, out_cb);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(in_cb);
        copy_tile(in_cb, 0, 0);
        transpose_wh_dest_init_short<is_32bit != 0>();
        transpose_wh_dest<is_32bit != 0>(0);
        tile_regs_commit();
    }

    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);
    cb_pop_front(in_cb, 2);
}
