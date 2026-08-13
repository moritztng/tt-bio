// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fourier-slice projection, stage 2 compute: one 1D affine resampling pass, tile-native.
//
// The pass computes out[r, u] = src(A*u + B*r + C) by linear interpolation. Splitting the source
// position into an integer and a fractional part,
//     p(r, u) = A*u + s(r),   s(r) = B*r + C,   k0(r) = floor(s(r)),   g(r) = s(r) - k0(r)
// the reader applies k0(r) as a per-row byte offset, so what reaches this kernel is
// srcp[r, j] = src[r, j + k0(r)] and the wanted position is q(r, u) = A*u + g(r) with g in [0, 1).
//
// ONE interpolation, not two. The obvious factorisation -- resample at A*u with a fixed matrix, then
// shift by g(r) -- cascades two linear interpolations and is a different, smoother interpolant than
// the one S2 part 1b measured at 1.23-1.27x RELION trilinear, so that accuracy number would not apply
// to this kernel. Instead the integer part of q is carried exactly:
//     floor(q) = floor(A*u) + M,   M = [frac(A*u) + g(r) >= 1] in {0, 1}
//     frac(q)  = frac(A*u) + g(r) - M
// which is a single linear interpolation between source samples floor(A*u) + M and + M + 1. Three
// fixed selection matrices supply the candidates:
//     T0 = srcp . P0   samples at floor(A*u)
//     T1 = srcp . P1   samples at floor(A*u) + 1
//     T2 = srcp . P2   samples at floor(A*u) + 2
//     base = lerp(T0, T1, M),  next = lerp(T1, T2, M),  out = lerp(base, next, frac(q))
// P0, P1, P2 are r-independent precisely because k0(r) is already in the reader's offset, and they are
// fixed for the whole orientation, so they are loaded once and stay L1-resident. Nothing is indexed
// per element and there is no gather.
//
// Exactness is free here: the pass is reader-bound (S1e, 1338.6 ns per output tile), so the extra
// matmuls and lerps come out of compute headroom that would otherwise idle.
//
// NOTE the reader can only apply offsets at 8-element granularity from an L1 source (16 B; 32
// elements from DRAM) -- measured in projprobe/fslice_align.py. This kernel therefore covers the
// shear family whose per-row integer offset is a multiple of 8. The residual 0..7 shift needs a wider
// candidate set than the three here and is not implemented.
//
// `mode` selects the output so a failure can be attributed rather than guessed at:
//   0 tilize only   1 full pass   2 emit T0 (selection matmul)   3 emit the weight tile w
//   4 full pass with M and frac(q) precomputed on the host
//
// Mode 4 is the optimisation the attribution asked for. Differencing the modes at 130 cores put
// the six SFPU ops at 1694.7 ns of a 2936.4 ns output tile -- 58% -- against roughly 90 ns for the
// six matmuls and the broadcast. But M and frac(q) are functions of g(r) and frac(A*u), and BOTH
// are known on the host, so building them on the device with a broadcast add, a copy, a floor and
// a frac is work that need not happen at all. Mode 4 reads them as two ready tiles and keeps only
// the three lerps, trading four device ops for two cheap FPU copies.
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
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_til = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(2);
    constexpr uint32_t cb_frac = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t src_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t mode = get_compile_time_arg_val(6);

    const uint32_t nblocks = get_arg_val<uint32_t>(0);
    constexpr uint32_t one = 1;

    binary_op_init_common(cb_src, cb_sel, cb_out);

    if constexpr (mode == 0) {
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

    cb_wait_front(cb_sel, 3 * src_tiles);
    cb_wait_front(cb_frac, 2);

    for (uint32_t b = 0; b < nblocks; ++b) {
        cb_wait_front(cb_src, src_tiles);

        tilize_init_short_with_dt(cb_out, cb_src, src_tiles, cb_til);
        cb_reserve_back(cb_til, src_tiles);
        tilize_block(cb_src, src_tiles, cb_til);
        cb_push_back(cb_til, src_tiles);
        tilize_uninit_with_dt(cb_src, cb_out, cb_til);
        cb_wait_front(cb_til, src_tiles);

        cb_reserve_back(cb_out, one);
        tile_regs_acquire();

        // T0, T1, T2: three r-independent selections of the 64-wide window. The matmul contracts over
        // the source index, which is what makes the selection native rather than a gather.
        mm_init(cb_til, cb_sel, cb_out, 0);
        for (uint32_t k = 0; k < src_tiles; ++k) {
            matmul_tiles(cb_til, cb_sel, k, k, DST_T0);
            matmul_tiles(cb_til, cb_sel, k, src_tiles + k, DST_T1);
            matmul_tiles(cb_til, cb_sel, k, 2 * src_tiles + k, DST_T2);
        }

        // w(r, u) = g(r) + frac(A*u). add_tiles_bcast_rows broadcasts ROW 0 of its SECOND operand, so
        // the per-row tile must be first and the per-column tile second. cb_frac tile 1 holds g(r)
        // replicated across the columns; tile 0 holds frac(A*u) in row 0.
        if constexpr (mode != 4) {
            add_bcast_rows_init_short(cb_frac, cb_frac);
            add_tiles_bcast_rows(cb_frac, cb_frac, 1, 0, DST_W);
        }

        if constexpr (mode == 3) {
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(DST_W, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, one);
            cb_pop_front(cb_til, src_tiles);
            cb_pop_front(cb_src, src_tiles);
            continue;
        }
        if constexpr (mode == 2) {
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(DST_T0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, one);
            cb_pop_front(cb_til, src_tiles);
            cb_pop_front(cb_src, src_tiles);
            continue;
        }

        if constexpr (mode == 4) {
            // cb_frac already holds frac(q) in tile 0 and M in tile 1, so nothing has to be derived.
            copy_tile_to_dst_init_short(cb_frac);
            copy_tile(cb_frac, 0, DST_W);
            copy_tile(cb_frac, 1, DST_M);
        } else {
            // M = floor(w) in {0, 1}; frac(q) = w - M = frac(w). copy_dest_values is (in, out), so the
            // weight is the SOURCE and M the destination -- passing these the other way round
            // overwrites w with an uninitialised register and makes the result independent of g(r).
            copy_dest_values(DST_W, DST_M);
            rounding_op_tile_init();
            floor_tile(DST_M);
            frac_tile(DST_W);
        }

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
