// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_mix compute: AF2's column attention over two MSA rows, from the projected q, k, v to the
// attended value, for one head of one token tile row (a head is 32 channels, one tile):
//
//   kd = k0 - k1                      vd = v0 - v1
//   gap_s = scale * rowsum(q_s * kd) + bias          (s = 0, 1; one number a token and head)
//   out_s = v1 + sigmoid(gap_s) * vd
//
// With two keys the softmax is a sigmoid of the logit gap, so nothing here leaves the tile. The row
// sum is a matmul with a constant tile full of `scale` (const tile 0), which also spreads it over
// the head's 32 columns; the bias column is spread the same way by a tile whose first row is ones
// (const tile 1). Everything else is the SFPU in a float32 DST, rounded once at the pack. The
// products q_s * kd go through a float32 CB because the FPU reads its operands from CBs.
//
// c_0 holds a unit's k0, k1, q0, q1, v0, v1 in that order.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#include "unit.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_bias = 1, cb_const = 2, cb_out = 16, cb_mid = 24;
    constexpr uint32_t BIAS = get_compile_time_arg_val(0);
    uint32_t num_units;
    {
        constexpr uint32_t G = 1;
        constexpr uint32_t COMPACT = get_compile_time_arg_val(G);
        if constexpr (COMPACT) {
            num_units = genq::slice<get_compile_time_arg_val(G + 1), get_compile_time_arg_val(G + 2),
                                    get_compile_time_arg_val(G + 3), get_compile_time_arg_val(G + 4),
                                    get_compile_time_arg_val(G + 5), get_compile_time_arg_val(G + 6),
                                    get_compile_time_arg_val(G + 7)>(
                            get_absolute_logical_x(), get_absolute_logical_y()).num;
        } else {
            num_units = get_arg_val<uint32_t>(0);
        }
    }

    binary_op_init_common(cb_in, cb_const, cb_out);
    cb_wait_front(cb_const, 2);
    for (uint32_t u = 0; u < num_units; ++u) {
        cb_wait_front(cb_in, 6);
        if constexpr (BIAS) {
            cb_wait_front(cb_bias, 1);
        }

        // q_s * kd, to the float32 CB
        cb_reserve_back(cb_mid, 2);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_in);
        copy_tile_to_dst_init_short(cb_in);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_in, i, i);
        }
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        mul_binary_tile_init();
        mul_binary_tile(2, 0, 2);
        mul_binary_tile(3, 0, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_mid);
        pack_tile(2, cb_mid);
        pack_tile(3, cb_mid);
        tile_regs_release();
        cb_push_back(cb_mid, 2);

        cb_wait_front(cb_mid, 2);
        cb_reserve_back(cb_out, 2);
        tile_regs_acquire();
        reconfig_data_format(cb_const, cb_mid);   // srcA takes in1 (the constant), srcB in0
        mm_init_short(cb_mid, cb_const, 0);
        matmul_tiles(cb_mid, cb_const, 0, 0, 0);
        matmul_tiles(cb_mid, cb_const, 1, 0, 1);
        if constexpr (BIAS) {
            reconfig_data_format(cb_const, cb_bias);
            mm_init_short(cb_bias, cb_const, 0);
            matmul_tiles(cb_bias, cb_const, 0, 1, 0);
            matmul_tiles(cb_bias, cb_const, 0, 1, 1);
        }
        reconfig_data_format_srca(cb_in);
        copy_tile_to_dst_init_short(cb_in);
        copy_tile(cb_in, 4, 2);
        copy_tile(cb_in, 5, 3);
        sigmoid_tile_init();
        sigmoid_tile(0);
        sigmoid_tile(1);
        sub_binary_tile_init();
        sub_binary_tile(2, 3, 2);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 3, 0);
        add_binary_tile(1, 3, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        pack_tile(1, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 2);

        cb_pop_front(cb_mid, 2);
        cb_pop_front(cb_in, 6);
        if constexpr (BIAS) {
            cb_pop_front(cb_bias, 1);
        }
    }
}
