// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 1's adjoint: V'[X, Y, p] = mask_p[X, Y] * W'[X, Y], for every p in the band.
//
// The forward is W = sum_p mask_p * V_p, so the adjoint applies the same masks to the scattered value
// and writes each product to its own plane. Section 5 called this "spreads one value into 2 z-planes
// weighted by the same mask" -- 2 planes carry weight per CELL, but the band a whole 32x32 tile touches
// is the same ~28 the forward contracts, so the tile-level work is the full band either way.
//
// Each volume tile is written by exactly ONE W tile within a direction, because W(X,Y) carries the
// (X,Y) index through unchanged -- the same fact that means the forward has no read reuse to exploit.
// So there is nothing to accumulate here across W tiles, and every product is written once.
//
// DST holds 8 tiles, so the band is processed in chunks of 8. mul_tiles ACCUMULATES into DST, so each
// slot must be written by exactly one multiply per acquire -- which is why j indexes a distinct slot.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"

#define CHUNK 8

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t nplane = get_compile_time_arg_val(3);

    const uint32_t nblocks = get_arg_val<uint32_t>(0);

    binary_op_init_common(cb_w, cb_mask, cb_out);
    cb_wait_front(cb_mask, nplane);

    for (uint32_t b = 0; b < nblocks; ++b) {
        cb_wait_front(cb_w, 1);
        cb_reserve_back(cb_out, nplane);
        for (uint32_t base = 0; base < nplane; base += CHUNK) {
            uint32_t n = nplane - base;
            if (n > CHUNK) {
                n = CHUNK;
            }
            tile_regs_acquire();
            mul_tiles_init(cb_w, cb_mask);
            for (uint32_t j = 0; j < n; ++j) {
                mul_tiles(cb_w, cb_mask, 0, base + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
        }
        cb_push_back(cb_out, nplane);
        cb_pop_front(cb_w, 1);
    }
}
