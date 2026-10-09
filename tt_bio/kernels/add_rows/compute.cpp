// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_rows compute: out = z + blk, FPU add_tiles into a 32-bit DEST, packed to bfloat16. That is
// the arithmetic `ttnn.add` does on two bfloat16 operands (rne_add/compute_rne_add.cpp measures
// the two equal element for element), so the in-place sum writes the bytes `add_` would.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    const uint32_t n = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_z = 0, cb_b = 1, cb_out = 16;

    binary_op_init_common(cb_z, cb_b, cb_out);
    add_tiles_init(cb_z, cb_b);
    for (uint32_t i = 0; i < n; i += GRAN) {
        const uint32_t k_n = (n - i < GRAN) ? (n - i) : GRAN;
        cb_wait_front(cb_z, k_n);
        cb_wait_front(cb_b, k_n);
        cb_reserve_back(cb_out, k_n);
        tile_regs_acquire();
        for (uint32_t j = 0; j < k_n; ++j) {
            add_tiles(cb_z, cb_b, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < k_n; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_pop_front(cb_z, k_n);
        cb_pop_front(cb_b, k_n);
        cb_push_back(cb_out, k_n);
    }
}
