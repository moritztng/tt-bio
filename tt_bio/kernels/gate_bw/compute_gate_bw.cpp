// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gate_bw compute: the backward of `o * sigmoid(g)`, per tile,
//   s  = sigmoid(g)
//   do = d * s
//   dg = d * o * (s - s * s)
// `reblock_permute_gated_bw`'s DST sequence without its transpose. The composed VJP is seven
// ops (sigmoid, two multiplies, then `ttnn.sigmoid_bw`, itself sigmoid/rsub/two multiplies), each
// rounded to the cotangent's dtype in DRAM; this is one float32 DST acquire, rounded at the pack.
// Every operand is copied straight into DST (`UnpackToDestFp32` on a float32 cotangent, set by
// the host), so a float32 d is not narrowed to SrcA's 19 bits.
//
// DST slots, four float32 tiles: 0 = d then s*s, 1 = s then s(1-s), 2 = o then dg, 3 = do.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#include "../genq/genq_split.h"

void kernel_main() {
    constexpr uint32_t d_cb = get_compile_time_arg_val(0);
    constexpr uint32_t o_cb = get_compile_time_arg_val(1);
    constexpr uint32_t g_cb = get_compile_time_arg_val(2);
    constexpr uint32_t do_cb = get_compile_time_arg_val(3);
    constexpr uint32_t dg_cb = get_compile_time_arg_val(4);
    constexpr uint32_t G = 5;
    constexpr uint32_t COMPACT = get_compile_time_arg_val(G);
    uint32_t num_tiles;
    if constexpr (COMPACT) {
        num_tiles = genq::slice<get_compile_time_arg_val(G + 1), get_compile_time_arg_val(G + 2),
                                get_compile_time_arg_val(G + 3), get_compile_time_arg_val(G + 4),
                                get_compile_time_arg_val(G + 5), get_compile_time_arg_val(G + 6),
                                get_compile_time_arg_val(G + 7)>(
                        get_absolute_logical_x(), get_absolute_logical_y()).num;
    } else {
        num_tiles = get_arg_val<uint32_t>(0);
    }

    binary_op_init_common(d_cb, g_cb, do_cb);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(d_cb, 1);
        cb_wait_front(o_cb, 1);
        cb_wait_front(g_cb, 1);
        cb_reserve_back(do_cb, 1);
        cb_reserve_back(dg_cb, 1);

        tile_regs_acquire();
        // `_with_dt` reconfigures the unpacker only where the formats differ (a float32 d).
        copy_tile_to_dst_init_short_with_dt(o_cb, d_cb);
        copy_tile(d_cb, 0, 0);
        copy_tile_to_dst_init_short_with_dt(d_cb, g_cb);
        copy_tile(g_cb, 0, 1);
        copy_tile_to_dst_init_short_with_dt(g_cb, o_cb);
        copy_tile(o_cb, 0, 2);
        sigmoid_tile_init();
        sigmoid_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 3);  // do = d * s
        mul_binary_tile(2, 0, 2);  // o * d
        mul_binary_tile(1, 1, 0);  // s * s
        sub_binary_tile_init();
        sub_binary_tile(1, 0, 1);  // s (1 - s)
        mul_binary_tile_init();
        mul_binary_tile(2, 1, 2);  // dg
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(do_cb);
        pack_tile(3, do_cb);
        pack_reconfig_data_format(dg_cb);
        pack_tile(2, dg_cb);
        tile_regs_release();

        cb_pop_front(d_cb, 1);
        cb_pop_front(o_cb, 1);
        cb_pop_front(g_cb, 1);
        cb_push_back(do_cb, 1);
        cb_push_back(dg_cb, 1);
    }
}
