// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// lead_sum compute: per output tile, acc = sum of its G input tiles, accumulated with the SFPU add
// in a float32 DST (the input is unpacked straight to DST, `UnpackToDestFp32`), packed once.
// DST slots: 0 = accumulator, 1 = the next tile.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

#include "../genq/genq_split.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t C = 3;
    constexpr uint32_t COMPACT = get_compile_time_arg_val(C);
    uint32_t num_tiles;
    if constexpr (COMPACT) {
        num_tiles = genq::slice<get_compile_time_arg_val(C + 1), get_compile_time_arg_val(C + 2),
                                get_compile_time_arg_val(C + 3), get_compile_time_arg_val(C + 4),
                                get_compile_time_arg_val(C + 5), get_compile_time_arg_val(C + 6),
                                get_compile_time_arg_val(C + 7)>(
                        get_absolute_logical_x(), get_absolute_logical_y()).num;
    } else {
        num_tiles = get_arg_val<uint32_t>(0);
    }

    binary_op_init_common(in_cb, in_cb, out_cb);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_reserve_back(out_cb, 1);
        tile_regs_acquire();
        cb_wait_front(in_cb, 1);
        copy_tile_to_dst_init_short(in_cb);
        copy_tile(in_cb, 0, 0);
        cb_pop_front(in_cb, 1);
        for (uint32_t k = 1; k < G; ++k) {
            cb_wait_front(in_cb, 1);
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, 0, 1);
            cb_pop_front(in_cb, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_push_back(out_cb, 1);
    }
}
