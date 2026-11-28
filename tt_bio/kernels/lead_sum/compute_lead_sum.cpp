// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// lead_sum compute: per output tile, acc = sum of its G input tiles, accumulated with the SFPU add
// in a float32 DST (the input is unpacked straight to DST, `UnpackToDestFp32`), packed once.
// DST slots: 0 = accumulator, 1 = the next tile. With PAIR the adds form a pairwise tree instead of
// a chain: the DST is a stack of partial sums, slab k lands in slot popcount(k) and every trailing
// one bit of k folds the top two slots, so at most log2(G) + 1 slots are live. The reader hands
// the slabs over bit-reversed, which makes this the tree that adds slab i to slab i + G/2,
// the order of a halving slice/add loop on the host.
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
    constexpr uint32_t PAIR = get_compile_time_arg_val(C + 8);
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
            uint32_t top = PAIR ? static_cast<uint32_t>(__builtin_popcount(k)) : 1;
            cb_wait_front(in_cb, 1);
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, 0, top);
            cb_pop_front(in_cb, 1);
            add_binary_tile_init();
            for (uint32_t c = PAIR ? k : 1; c & 1; c >>= 1, --top) {
                add_binary_tile(top - 1, top, top - 1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_push_back(out_cb, 1);
    }
}
