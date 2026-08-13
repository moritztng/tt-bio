// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Backprojection: the adjoint of the 1D affine resampling pass.
//
// Section 5 predicted the whole structure before any kernel existed. The forward pass is
//     out[r, u] = sum_d c_d(r, u) * (srcp . P_d)[r, u]
// so its adjoint scatters a slice back into the source lattice as
//     src'[r, j] = sum_d (c_d(r, u) * y[r, u]) . P_d^T
// Every piece transposes exactly as section 5 said it would: the shared selection matrices become their
// own transposes, the per-element coefficients are unchanged, and the per-row offset moves from the
// reader to the writer.
//
// The coefficients have to be applied BEFORE the contraction, because c_d depends on u and u is what
// the matmul sums over. Since matmul reads both operands from circular buffers, each weighted slice is
// packed out to a staging buffer and read back -- the one structural cost the forward pass does not pay.
//
// Accumulation across contributions is FREE: matmul_tiles accumulates into DST (section 24.1), so every
// orientation touching this volume tile adds into the same registers with no read-modify-write, no
// atomics, and no cross-core reduction. That is section 5's central claim, and it is the reason
// backprojection needs no ttnn.scatter -- no O(destination volume) pass and no fp32 refusal.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

#define DST_W 0

void kernel_main() {
    constexpr uint32_t cb_y = get_compile_time_arg_val(0);      // the slice tile being scattered
    constexpr uint32_t cb_coef = get_compile_time_arg_val(1);   // c0, c1, c2
    constexpr uint32_t cb_selt = get_compile_time_arg_val(2);   // P0^T, P1^T, P2^T
    constexpr uint32_t cb_mid = get_compile_time_arg_val(3);    // staging for the weighted slice
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(5); // 2 -- the 64-wide source window
    constexpr uint32_t ncontrib = get_compile_time_arg_val(6);  // orientations accumulated per volume tile

    const uint32_t nblocks = get_arg_val<uint32_t>(0);
    constexpr uint32_t one = 1;

    binary_op_init_common(cb_y, cb_coef, cb_out);
    cb_wait_front(cb_coef, 3);
    cb_wait_front(cb_selt, 3 * out_tiles);

    for (uint32_t b = 0; b < nblocks; ++b) {
        // Stage every weighted slice first. The coefficient multiply needs its own acquire because it
        // ends in a pack, and acquires cannot nest -- so the accumulation cannot span them.
        for (uint32_t n = 0; n < ncontrib; ++n) {
            cb_wait_front(cb_y, one);
            for (uint32_t d = 0; d < 3; ++d) {
                cb_reserve_back(cb_mid, one);
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_y);
                copy_tile(cb_y, 0, DST_W);
                binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_coef);
                binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_coef, d, DST_W);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_W, cb_mid);
                tile_regs_release();
                cb_push_back(cb_mid, one);
            }
            cb_pop_front(cb_y, one);
        }

        // ONE acquire for every contribution, so matmul_tiles accumulates them all into DST with no
        // read-modify-write, no atomics and no cross-core reduction. Section 5's central claim.
        cb_wait_front(cb_mid, 3 * ncontrib);
        cb_reserve_back(cb_out, out_tiles);
        tile_regs_acquire();
        mm_init(cb_mid, cb_selt, cb_out, 0);
        for (uint32_t n = 0; n < ncontrib; ++n) {
            for (uint32_t d = 0; d < 3; ++d) {
                for (uint32_t k = 0; k < out_tiles; ++k) {
                    matmul_tiles(cb_mid, cb_selt, n * 3 + d, d * out_tiles + k, k);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t k = 0; k < out_tiles; ++k) {
            pack_tile(k, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, out_tiles);
        cb_pop_front(cb_mid, 3 * ncontrib);
    }
}
