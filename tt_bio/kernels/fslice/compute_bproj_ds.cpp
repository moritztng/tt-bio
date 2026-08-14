// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 2's adjoint, DESTINATION-STATIONARY: one W tile, every slice that contributes to it.
//
// compute_bproj.cpp runs the same arithmetic source-stationary -- fix the slice tile y, produce the
// 64-wide W window, write it at 32 per-row offsets. That form needs a scattered write, and because a
// NoC write overwrites rather than adds it needs a scattered read-modify-write. Batching
// contributions in DST does not amortise it either: the batched contributions are different in-plane
// rotations psi, and psi is exactly what sets the per-row offset, so one packed tile would have to
// land at as many different sets of 32 offsets as there are contributions.
//
// Fixing the DESTINATION moves the per-row offset back to the reader. The forward's own reader
// assembles 32 rows at 32 offsets out of an L1-resident plane; here it does the same thing out of an
// L1-resident slice -- same transaction, same count, same kernel. The write becomes one bulk page.
//
//     src'[r, j] = sum_d ( c_d(r, u) * y[r, u] ) . P_d^T[u, j]
//
// The coefficients apply BEFORE the contraction, because c_d depends on u and u is what the matmul
// sums over, while matmul reads both operands from circular buffers. So each weighted window tile is
// packed to a staging CB and read back. That round trip is the one structural cost the forward does
// not pay: the forward's c_d depends on the OUTPUT index, so it multiplies inline in DST.
//
// THE STAGING CB IS WHAT BOUNDS THE ACCUMULATION DEPTH. A single acquire can only matmul tiles that
// are already in cb_mid, so "one acquire for all contributions" needs 3 * src_tiles * ncontrib tiles
// resident. Beyond that the contributions are chunked and the running sum crosses cb_acc.
//
// TWO RULES THIS KERNEL EXISTS TO ENCODE, both measured in projprobe/bproj_s2_diag.py:
//
//   1. mm_init is not a lightweight reconfigure. It calls llk_math_pack_sync_init and
//      llk_pack_dest_init, which reset the dest section base and the dest offset, so ANY value
//      already in DST is discarded. Calling it inside tile_regs_acquire() after copy_tile has seeded
//      DST with the running sum threw the seed away, and the landed contribution count did not move
//      at all between 1, 2 and 3 chunks. It is hoisted out of the acquire here and the seed is
//      followed by mm_init_short_with_dt, which only re-points the unpacker and the math engine.
//   2. With fp32_dest_acc_en on, EVERY CB this kernel packs into must carry the SAME data format.
//      cb_mid at bf16 against cb_acc/cb_out at fp32 silently dropped the first tile packed after
//      each mm_init -- exactly one contribution per acquire, 12.5 % at chunk 8 -- and
//      pack_reconfig_data_format did not cover it. So DST accumulates in fp32 and L1 stays bf16.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"

#define DST_W 0

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_til = get_compile_time_arg_val(1);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(2);
    constexpr uint32_t cb_selt = get_compile_time_arg_val(3);
    constexpr uint32_t cb_mid = get_compile_time_arg_val(4);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t src_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t ncontrib = get_compile_time_arg_val(8);
    constexpr uint32_t chunk = get_compile_time_arg_val(9);

    constexpr uint32_t nmid = 3 * src_tiles;
    constexpr uint32_t nchunk = ncontrib / chunk;

    const uint32_t nblocks = get_arg_val<uint32_t>(0);
    constexpr uint32_t one = 1;

    binary_op_init_common(cb_src, cb_coef, cb_out);
    cb_wait_front(cb_coef, 3);
    cb_wait_front(cb_selt, nmid);

    for (uint32_t b = 0; b < nblocks; ++b) {
        for (uint32_t c = 0; c < nchunk; ++c) {
            for (uint32_t n = 0; n < chunk; ++n) {
                cb_wait_front(cb_src, src_tiles);
                tilize_init(cb_src, src_tiles, cb_til);
                cb_reserve_back(cb_til, src_tiles);
                tilize_block(cb_src, src_tiles, cb_til);
                cb_push_back(cb_til, src_tiles);
                tilize_uninit(cb_src, cb_til);
                cb_wait_front(cb_til, src_tiles);

                for (uint32_t d = 0; d < 3; ++d) {
                    for (uint32_t k = 0; k < src_tiles; ++k) {
                        cb_reserve_back(cb_mid, one);
                        tile_regs_acquire();
                        copy_tile_to_dst_init_short(cb_til);
                        copy_tile(cb_til, k, DST_W);
                        binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_coef);
                        binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_coef, d, DST_W);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_reconfig_data_format(cb_mid);
                        pack_tile(DST_W, cb_mid);
                        tile_regs_release();
                        cb_push_back(cb_mid, one);
                    }
                }
                cb_pop_front(cb_til, src_tiles);
                cb_pop_front(cb_src, src_tiles);
            }

            const bool last = (c + 1 == nchunk);
            const uint32_t cb_dst = last ? cb_out : cb_acc;
            cb_wait_front(cb_mid, nmid * chunk);
            cb_reserve_back(cb_dst, one);
            if (c) {
                cb_wait_front(cb_acc, one);
            }
            // OUTSIDE the acquire: this resets the dest section, so nothing in DST survives it.
            mm_init(cb_mid, cb_selt, cb_dst, 0);
            tile_regs_acquire();
            if (c) {
                copy_tile_to_dst_init_short_with_dt(cb_selt, cb_acc);
                copy_tile(cb_acc, 0, DST_W);
                mm_init_short_with_dt(cb_mid, cb_selt, cb_acc, 0);
            }
            for (uint32_t n = 0; n < chunk; ++n) {
                for (uint32_t i = 0; i < nmid; ++i) {
                    matmul_tiles(cb_mid, cb_selt, n * nmid + i, i, DST_W);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_dst);
            pack_tile(DST_W, cb_dst);
            tile_regs_release();
            if (c) {
                cb_pop_front(cb_acc, one);
            }
            cb_push_back(cb_dst, one);
            cb_pop_front(cb_mid, nmid * chunk);
        }
    }
}
