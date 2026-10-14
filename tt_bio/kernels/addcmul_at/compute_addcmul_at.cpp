// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// addcmul_at compute: `out = x + y * z`, the product on the FPU into DEST, then x added on the FPU
// with DEST reused as srcA (the trimul_tail_res idiom). Same function as `ttnn.addcmul(x, y, z)`;
// the product is truncated to x's format on its way back into srcA, so it is not bit-exact with it.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = 0, cb_y = 1, cb_z = 2, cb_out = 16;

    binary_op_init_common(cb_y, cb_z, cb_out);
    for (uint32_t i = 0; i < num_tiles; i += GRAN) {
        const uint32_t n = (num_tiles - i < GRAN) ? (num_tiles - i) : GRAN;
        cb_wait_front(cb_x, n);
        cb_wait_front(cb_y, n);
        cb_wait_front(cb_z, n);
        cb_reserve_back(cb_out, n);
        tile_regs_acquire();
        mul_tiles_init(cb_y, cb_z);
        for (uint32_t j = 0; j < n; ++j) mul_tiles(cb_y, cb_z, j, j, j);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x);
        for (uint32_t j = 0; j < n; ++j)
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x, j, j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) pack_tile(j, cb_out);
        tile_regs_release();
        cb_pop_front(cb_x, n);
        cb_pop_front(cb_y, n);
        cb_pop_front(cb_z, n);
        cb_push_back(cb_out, n);
    }
}
