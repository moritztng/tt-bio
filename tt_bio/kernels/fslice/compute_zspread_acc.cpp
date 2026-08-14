// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 1's adjoint, DESTINATION-STATIONARY: one volume tile, every contribution to it.
//
//     V'[X, Y, z] = sum_{d : band_d covers z}  mask_{z - z0_d}[X, Y] * W_d[X, Y]
//
// compute_zspread.cpp runs this the other way round -- one W tile spread across its band, nplane
// tiles written per W tile. That form has to write each volume tile once per contributing direction,
// and since a NoC write overwrites rather than adds, every write after the first is a read-modify-
// write. Fixing the DESTINATION instead makes all ~84 contributions land in one DST slot under one
// acquire, and the volume tile is written exactly once per iteration with a bulk page write.
//
// That is the whole no-scatter argument, and it is the same mechanism the forward's z-collapse uses:
// mul_tiles ACCUMULATES into DST, so the sum over the contributions falls out of the multiplies at
// no cost. compute_zspread.cpp's comment warns that each DST slot must be written by exactly one
// multiply per acquire -- that warning is for a kernel that wants nplane INDEPENDENT products. Here
// the accumulation is the point, so every multiply targets slot 0 deliberately.
//
// THE SLIDING WINDOW. For a fixed (X, Y) each direction's band is contiguous in z, so sorting the
// directions by where their band starts and stepping z by one makes the active set a window: nstep
// directions enter and nstep leave per step, and a W tile that has been read stays resident for the
// nplane steps it is active. That is why the adjoint reads 5.33 W tiles per slice where the forward
// reads 149.3 volume tiles -- one read serves nplane edges instead of one.
//
// The CB is in age order, because the reader pushes at the back and this kernel pops from the front.
// So CB index i holds a direction that entered (nwin-1-i)/nstep steps ago, and that integer IS its
// band position, which is the mask index. No per-contribution mask read: the nplane masks are the
// forward's own, resident, loaded once.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t nplane = get_compile_time_arg_val(3);   // band depth, S3's mean is 28
    constexpr uint32_t nstep = get_compile_time_arg_val(4);    // directions entering per z step
    constexpr uint32_t nwin = nplane * nstep;                  // contributions live at any z

    const uint32_t nblocks = get_arg_val<uint32_t>(0);

    binary_op_init_common(cb_w, cb_mask, cb_out);
    cb_wait_front(cb_mask, nplane);

    for (uint32_t b = 0; b < nblocks; ++b) {
        cb_wait_front(cb_w, nwin);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        mul_tiles_init(cb_w, cb_mask);
        // ONE acquire, every contribution accumulating into DST slot 0. No read-modify-write, no
        // atomics, no cross-core reduction, and no scattered write.
        for (uint32_t i = 0; i < nwin; ++i) {
            mul_tiles(cb_w, cb_mask, i, (nwin - 1 - i) / nstep, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_w, nstep);   // only the expired directions leave: the window slides
    }
}
