// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute_gated BACKWARD compute: per output tile,
//   da' = transpose_wh(gathered da tile)        (the inverse move's last step)
//   s   = sigmoid(g)
//   dp  = da' * s
//   dg  = da' * p * (s - s * s)
// all in one float32 DST acquire, so nothing between the cotangent and the two gradients is rounded
// to bfloat16. The composed path this replaces (the move back, then sigmoid, rsub, and five
// multiplies, each written to DRAM in bfloat16) rounds six times; this rounds once, at the pack.
// The backward is graded against float64, not against that chain, so the extra precision is the
// point and not a deviation.
//
// DST slots, four float32 tiles: 0 = da' then s*s, 1 = s then s(1-s), 2 = p then dg, 3 = dp.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    constexpr uint32_t da_cb = get_compile_time_arg_val(0);  // c_0, gathered da tiles (reader)
    constexpr uint32_t p_cb = get_compile_time_arg_val(1);
    constexpr uint32_t g_cb = get_compile_time_arg_val(2);
    constexpr uint32_t dp_cb = get_compile_time_arg_val(3);
    constexpr uint32_t dg_cb = get_compile_time_arg_val(4);
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    transpose_wh_init(da_cb, dp_cb);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(da_cb, 1);
        cb_wait_front(p_cb, 1);
        cb_wait_front(g_cb, 1);
        cb_reserve_back(dp_cb, 1);
        cb_reserve_back(dg_cb, 1);

        tile_regs_acquire();
        transpose_wh_init_short(da_cb);
        transpose_wh_tile(da_cb, 0, 0);
        copy_tile_to_dst_init_short(g_cb);
        copy_tile(g_cb, 0, 1);
        copy_tile_to_dst_init_short(p_cb);
        copy_tile(p_cb, 0, 2);
        sigmoid_tile_init();
        sigmoid_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 3);  // dp = da' * s
        mul_binary_tile(2, 0, 2);  // p * da'
        mul_binary_tile(1, 1, 0);  // s * s
        sub_binary_tile_init();
        sub_binary_tile(1, 0, 1);  // s (1 - s)
        mul_binary_tile_init();
        mul_binary_tile(2, 1, 2);  // dg
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(3, dp_cb);
        pack_tile(2, dg_cb);
        tile_regs_release();

        cb_pop_front(da_cb, 1);
        cb_pop_front(p_cb, 1);
        cb_pop_front(g_cb, 1);
        cb_push_back(dp_cb, 1);
        cb_push_back(dg_cb, 1);
    }
}
