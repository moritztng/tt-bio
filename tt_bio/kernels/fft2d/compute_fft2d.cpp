// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A fused 2D complex FFT on Blackhole, as a blocked matmul that never leaves L1.
//
// Why this shape. Screen S2b measured a 32x32x32 tile matmul at 26.42 ns against 60.89 ns for an
// elementwise tile multiply, so per flop the matrix engine is 148x more efficient than the eltwise
// engine, and E1 showed the FPU's precision cap costs 0.0000 A of resolution at gold-standard FSC.
// Together those remove every reason to build the transform out of eltwise butterflies. So the
// whole transform is matmuls: Y = F . X . F, blocked into 32x32 tiles, with F the DFT matrix.
//
// The image and the DFT matrix are both L1-resident for the whole transform, so DRAM sees one read
// and one write per image rather than the four a two-pass GPU FFT needs. That halving is the
// performance thesis of this task and it is what a GPU cannot do, because it has no equivalent of
// 1.5 MB of private SRAM per core.
//
// Complex arithmetic without a subtract. matmul_tiles only accumulates (DST += A.B), and the real
// part of a complex product needs Xr.Fr - Xi.Fi. Rather than negate on the fly, the host also
// supplies -Fi as a third block set, so every term is an accumulate:
//     out_r = Xr.Fr + Xi.(-Fi)
//     out_i = Xr.Fi + Xi.Fr
// F is a constant, so negating it costs nothing at runtime and buys a branch-free inner loop.
//
// Both accumulators live in DST at once (index 0 real, index 1 imaginary) and are packed together,
// so the 16 matmuls that produce one complex output tile pay a single pack.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t cb_f = get_compile_time_arg_val(0);     // Fr, Fi, -Fi interleaved by 3
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);     // Xr, Xi interleaved by 2
    constexpr uint32_t cb_t = get_compile_time_arg_val(2);     // intermediate, same interleave
    constexpr uint32_t cb_o = get_compile_time_arg_val(3);     // output, drained by the writer
    constexpr uint32_t NT = get_compile_time_arg_val(4);       // tiles per side, box / 32
    constexpr uint32_t NIMG = get_compile_time_arg_val(5);     // images per core per launch

    constexpr uint32_t NPOS = NT * NT;          // complex tile positions in one image
    constexpr uint32_t NTILE = 2 * NPOS;        // real tiles in one image

    // Block accessors. The interleave is the storage convention, not an algorithmic choice: it
    // keeps the reader a single linear stream and the host layout symmetric between input and
    // output.
    #define FR(m, j) (3 * ((m) * NT + (j)))
    #define FI(m, j) (3 * ((m) * NT + (j)) + 1)
    #define FN(m, j) (3 * ((m) * NT + (j)) + 2)
    #define XR(i, m) (2 * ((i) * NT + (m)))
    #define XI(i, m) (2 * ((i) * NT + (m)) + 1)

    cb_wait_front(cb_f, 3 * NPOS);              // the DFT matrix, read once per launch

    for (uint32_t img = 0; img < NIMG; ++img) {
        cb_wait_front(cb_x, NTILE);

        // Pass 1 -- transform along the last axis: T = X . F
        mm_init(cb_x, cb_f, cb_t, 0);
        for (uint32_t i = 0; i < NT; ++i) {
            for (uint32_t j = 0; j < NT; ++j) {
                cb_reserve_back(cb_t, 2);
                tile_regs_acquire();
                for (uint32_t m = 0; m < NT; ++m) {
                    matmul_tiles(cb_x, cb_f, XR(i, m), FR(m, j), 0);
                    matmul_tiles(cb_x, cb_f, XI(i, m), FN(m, j), 0);
                    matmul_tiles(cb_x, cb_f, XR(i, m), FI(m, j), 1);
                    matmul_tiles(cb_x, cb_f, XI(i, m), FR(m, j), 1);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_t);
                pack_tile(1, cb_t);
                tile_regs_release();
                cb_push_back(cb_t, 2);
            }
        }

        // Pass 2 -- transform along the first axis: Y = F . T. No DRAM round trip between the
        // passes; T never left L1. This is the whole point.
        cb_wait_front(cb_t, NTILE);
        mm_init(cb_f, cb_t, cb_o, 0);
        for (uint32_t i = 0; i < NT; ++i) {
            for (uint32_t j = 0; j < NT; ++j) {
                cb_reserve_back(cb_o, 2);
                tile_regs_acquire();
                for (uint32_t m = 0; m < NT; ++m) {
                    matmul_tiles(cb_f, cb_t, FR(i, m), XR(m, j), 0);
                    matmul_tiles(cb_f, cb_t, FN(i, m), XI(m, j), 0);
                    matmul_tiles(cb_f, cb_t, FI(i, m), XR(m, j), 1);
                    matmul_tiles(cb_f, cb_t, FR(i, m), XI(m, j), 1);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_o);
                pack_tile(1, cb_o);
                tile_regs_release();
                cb_push_back(cb_o, 2);
            }
        }
        cb_pop_front(cb_t, NTILE);
        cb_pop_front(cb_x, NTILE);
    }
}
