// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_transpose_add compute: per unit, sum = z + transposed u for 32 tiles, add_tiles into a 32-bit
// DEST packed to bfloat16 -- ttnn.add's arithmetic on two bfloat16 tensors (see add_rows/compute.cpp),
// so the result is the bytes pair_transpose + add_ wrote. The unit's inputs are popped before its
// last sums are pushed: the writer reads "all 32 sums in" as "the slot is free".
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    const uint32_t num = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_u = 16, cb_z = 3, cb_sum = 17;

    binary_op_init_common(cb_z, cb_u, cb_sum);
    add_tiles_init(cb_z, cb_u);
    for (uint32_t i = 0; i < num; ++i) {
        cb_wait_front(cb_u, 32);
        cb_wait_front(cb_z, 32);
        for (uint32_t q = 0; q < 32; q += GRAN) {
            cb_reserve_back(cb_sum, GRAN);
            tile_regs_acquire();
            for (uint32_t j = 0; j < GRAN; ++j) {
                add_tiles(cb_z, cb_u, q + j, q + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < GRAN; ++j) {
                pack_tile(j, cb_sum);
            }
            tile_regs_release();
            if (q + GRAN == 32) {
                cb_pop_front(cb_u, 32);
                cb_pop_front(cb_z, 32);
            }
            cb_push_back(cb_sum, GRAN);
        }
    }
}
