// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// addcmul_at compute: `out = x + y * z`, the product on the FPU into DEST, then x added on the FPU
// with DEST reused as srcA (the trimul_tail_res idiom). Same function as `ttnn.addcmul(x, y, z)`;
// the product is truncated to x's format on its way back into srcA, so it is not bit-exact with it.
// ADDCMUL_SFPU (fp32 operands): the FPU reads fp32 at tf32, so the three tiles are unpacked to DEST at
// fp32 and multiplied and added on the SFPU, one tile at a time, as ttnn's fp32 binary ops do.
// SIG_Z (`dit_cond_sigat`): `out = x + y * sigmoid(z)`, z being the conditioning logits. z goes to DEST, the
// sigmoid runs there on the SFPU, and DEST is reused as srcA for the product with y, then for the sum with x.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#ifdef ADDCMUL_SFPU
#include "api/compute/eltwise_binary_sfpu.h"
#endif
#include "api/compute/tile_move_copy.h"

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
#ifdef ADDCMUL_SFPU
        for (uint32_t j = 0; j < n; ++j) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_x);
            copy_tile(cb_x, j, 0);
            copy_tile_to_dst_init_short(cb_y);
            copy_tile(cb_y, j, 1);
            copy_tile_to_dst_init_short(cb_z);
            copy_tile(cb_z, j, 2);
#ifdef SIG_Z
            sigmoid_tile_init();
            sigmoid_tile(2);
#endif
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
#else
        tile_regs_acquire();
#ifdef SIG_Z
        copy_tile_to_dst_init_short(cb_z);
        for (uint32_t j = 0; j < n; ++j) copy_tile(cb_z, j, j);
        sigmoid_tile_init();
        for (uint32_t j = 0; j < n; ++j) sigmoid_tile(j);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_y);
        for (uint32_t j = 0; j < n; ++j)
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_y, j, j);
#else
        mul_tiles_init(cb_y, cb_z);
        for (uint32_t j = 0; j < n; ++j) mul_tiles(cb_y, cb_z, j, j, j);
#endif
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x);
        for (uint32_t j = 0; j < n; ++j)
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x, j, j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) pack_tile(j, cb_out);
        tile_regs_release();
#endif
        cb_pop_front(cb_x, n);
        cb_pop_front(cb_y, n);
        cb_pop_front(cb_z, n);
        cb_push_back(cb_out, n);
    }
}
