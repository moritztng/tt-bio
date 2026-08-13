// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 1 compute: the z-collapse, W[X, Y] = V[X, Y, a*X + b*Y] by linear interpolation along z.
//
// Every cell of a 32x32 (X, Y) tile draws from the two z-planes bracketing its own z, and z varies
// across the tile, so the band of planes the tile touches is wide (S3: mean 28.27) even though only two
// of them matter per cell. Written as a sum over the band,
//     W = sum_p mask_p * V_p
// with mask_p[X, Y] carrying the interpolation weight where plane p brackets that cell and zero
// elsewhere. The masks are functions of a, b and the lattice, so they are computed on the host and are
// fixed for the whole direction -- the same move that took stage 2's SFPU bill down in section 18.
//
// Two ops per plane: an FPU mul_tiles reading both operands from circular buffers, and an SFPU
// add_binary_tile to accumulate DST into DST. There is no FPU DST-to-DST add, which is what keeps this
// at two rather than one.
//
// This is amortised 96-fold in the real pipeline, because W depends only on the viewing direction while
// stage 2 applies the in-plane rotation, and there are 96 psi values per direction at healpix order 4.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

#define DST_ACC 0
#define DST_TMP 1
#define DST_ACC2 2

void kernel_main() {
    constexpr uint32_t cb_v = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t nplane = get_compile_time_arg_val(3);

    const uint32_t nblocks = get_arg_val<uint32_t>(0);
    constexpr uint32_t one = 1;

    binary_op_init_common(cb_v, cb_mask, cb_out);
    cb_wait_front(cb_mask, nplane);

    for (uint32_t b = 0; b < nblocks; ++b) {
        cb_wait_front(cb_v, nplane);
        cb_reserve_back(cb_out, one);
        tile_regs_acquire();

        // The accumulator PING-PONGS between two registers. add_binary_tile(a, b, out) with out equal
        // to one of its inputs does not accumulate correctly -- it was the first thing tried here and
        // it produced rel L2 2.9 at 8 planes and 16 at 28, growing with the plane count exactly as a
        // broken accumulation would.
        uint32_t acc = DST_ACC;
        // mul_tiles ACCUMULATES into DST rather than overwriting it, exactly as matmul_tiles does.
        // That was not obvious and it cost two wrong diagnoses here: an accumulator that ping-ponged
        // registers to avoid an in-place add (which was never the problem), and a mul-only arm that
        // looked broken only because it was compared against a single product rather than the running
        // sum it was actually computing.
        //
        // Exploited rather than worked around, it makes the whole z-collapse ONE op per plane with no
        // SFPU involvement at all: the sum over the band falls out of the multiplies themselves.
        mul_tiles_init(cb_v, cb_mask);
        for (uint32_t p = 0; p < nplane; ++p) {
            mul_tiles(cb_v, cb_mask, p, p, DST_ACC);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(DST_ACC, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, one);
        cb_pop_front(cb_v, nplane);
    }
}
