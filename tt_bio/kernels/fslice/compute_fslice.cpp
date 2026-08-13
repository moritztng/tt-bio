// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fourier-slice projection, stage 2 compute: one 1D affine resampling pass, tile-native.
//
// The pass computes out[r, u] = src(A*u + B*r + C) by linear interpolation. Splitting the source
// position into an integer and a fractional part,
//     p(r, u) = A*u + s(r),   s(r) = B*r + C,   k0(r) = floor(s(r)),   g(r) = s(r) - k0(r)
// the reader has already applied k0(r) as a per-row byte offset, so what reaches this kernel is
// srcp[r, j] = src[r, j + k0(r)] and the position wanted is q(r, u) = A*u + g(r) with g in [0, 1).
//
// ONE interpolation, not two. The obvious factorisation -- resample at A*u with a fixed matrix, then
// shift by g(r) -- cascades two linear interpolations and is a different, smoother interpolant than
// the one S2 part 1b measured at 1.23-1.27x RELION trilinear. Cascading would make that accuracy
// number not apply to this kernel. So instead the integer part of q is carried exactly:
//     floor(q) = floor(A*u) + M,   M = [frac(A*u) + g(r) >= 1] in {0, 1}
//     frac(q)  = frac(A*u) + g(r) - M
// giving a single linear interpolation between source samples floor(A*u) + M and + M + 1. Three fixed
// selection matrices supply the three candidate samples:
//     T0 = srcp . P0   samples at floor(A*u)
//     T1 = srcp . P1   samples at floor(A*u) + 1
//     T2 = srcp . P2   samples at floor(A*u) + 2
// then two lerps pick the M-dependent pair and a third does the interpolation:
//     base = lerp(T0, T1, M),  next = lerp(T1, T2, M),  out = lerp(base, next, frac(q))
// P0, P1 and P2 are r-independent because k0(r) is already in the reader's offset, and they are fixed
// for the whole orientation, so they are read once and stay L1-resident. Nothing is indexed per
// element and there is no gather.
//
// This costs more than the cascaded form -- 6 matmuls against 2, plus two extra lerps -- and it is
// free anyway: the pass is reader-bound at 1338.6 ns per output tile (S1e) against a compute budget
// this leaves well under it, so exactness is bought out of headroom that would otherwise idle.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"

#define DST_T0 0
#define DST_T1 1
#define DST_T2 2
#define DST_W 3
#define DST_M 4
#define DST_BASE 5
#define DST_NEXT 6
#define DST_OUT 7

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);     // row-major window from the reader
    constexpr uint32_t cb_til = get_compile_time_arg_val(1);     // tilized window, SRC_TILES wide
    constexpr uint32_t cb_sel = get_compile_time_arg_val(2);     // P0, P1, P2 selection matrices
    constexpr uint32_t cb_frac = get_compile_time_arg_val(3);    // frac(A*u) in row 0; g(r) in col 0
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t src_tiles = get_compile_time_arg_val(5);  // 2 -- the 64-wide window
    constexpr uint32_t mode = get_compile_time_arg_val(6);       // 0 = tilize only, 1 = full pass

    const uint32_t nblocks = get_arg_val<uint32_t>(0);
    constexpr uint32_t one = 1;

    binary_op_init_common(cb_src, cb_sel, cb_out);

    if constexpr (mode == 0) {
        // Plumbing arm: tilize the assembled window and emit it unchanged, so the reader's per-row
        // offsets and the row-major-to-tile conversion can be checked bit-exactly against the host
        // before any arithmetic is layered on top.
        tilize_init(cb_src, src_tiles, cb_out);
        for (uint32_t b = 0; b < nblocks; ++b) {
            cb_wait_front(cb_src, src_tiles);
            cb_reserve_back(cb_out, src_tiles);
            tilize_block(cb_src, src_tiles, cb_out);
            cb_push_back(cb_out, src_tiles);
            cb_pop_front(cb_src, src_tiles);
        }
        tilize_uninit(cb_src, cb_out);
        return;
    }

    // The three selection matrices and the two fraction vectors are fixed for the orientation and are
    // read once, before the block loop, rather than per output tile.
    cb_wait_front(cb_sel, 3 * src_tiles);
    cb_wait_front(cb_frac, 2);

    for (uint32_t b = 0; b < nblocks; ++b) {
        cb_wait_front(cb_src, src_tiles);

        // Row-major window -> tiles. tilize writes to a CB, so it goes to cb_til and comes back in.
        tilize_init_short_with_dt(cb_out, cb_src, src_tiles, cb_til);
        cb_reserve_back(cb_til, src_tiles);
        tilize_block(cb_src, src_tiles, cb_til);
        cb_push_back(cb_til, src_tiles);
        tilize_uninit_with_dt(cb_src, cb_out, cb_til);
        cb_wait_front(cb_til, src_tiles);

        cb_reserve_back(cb_out, one);
        tile_regs_acquire();

        // T0, T1, T2: three r-independent selections of the 64-wide window. Each is a sum over the
        // window's tiles, which is what a matmul contraction does natively.
        mm_init(cb_til, cb_sel, cb_out, 0);
        for (uint32_t k = 0; k < src_tiles; ++k) {
            matmul_tiles(cb_til, cb_sel, k, k, DST_T0);
            matmul_tiles(cb_til, cb_sel, k, src_tiles + k, DST_T1);
            matmul_tiles(cb_til, cb_sel, k, 2 * src_tiles + k, DST_T2);
        }

        // w(r, u) = frac(A*u) + g(r): a per-column vector broadcast down the rows plus a per-row
        // vector broadcast across the columns. Two native broadcast ops, no materialised weight tile.
        add_bcast_rows_init_short(cb_frac, cb_frac);
        add_tiles_bcast_rows(cb_frac, cb_frac, 0, 1, DST_W);

        // M = floor(w) in {0, 1}; frac(q) = w - M = frac(w).
        copy_dest_values_init();
        copy_dest_values(DST_M, DST_W);
        rounding_op_tile_init();
        floor_tile(DST_M);
        frac_tile(DST_W);

        // One interpolation: pick the M-dependent adjacent pair, then interpolate within it.
        lerp_tile_init();
        lerp_tile<DataFormat::Float16_b>(DST_T0, DST_T1, DST_M, DST_BASE);
        lerp_tile<DataFormat::Float16_b>(DST_T1, DST_T2, DST_M, DST_NEXT);
        lerp_tile<DataFormat::Float16_b>(DST_BASE, DST_NEXT, DST_W, DST_OUT);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(DST_OUT, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, one);

        cb_pop_front(cb_til, src_tiles);
        cb_pop_front(cb_src, src_tiles);
    }
}
