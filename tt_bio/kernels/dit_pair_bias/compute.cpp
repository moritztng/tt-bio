// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dit_pair_bias: every DiT block's SDPA mask straight from the pair tensor, head-first,
//   out[b, h, i, j] = (z[b, i, j, :] @ W[:, h]) * SCALE + mask[i, j],
// rows i >= I coming out as the (pre-padded) mask alone. Per unit (b, ib, jb) and head tile hb:
//   1. P_i = W^T[hb] @ z(i, jb)^T for the unit's 32 rows i (matmul with in1 transposed): tile i is [h x j].
//   2. The 32 P_i packed untilized as ONE row of 32 tiles into c_3, so c_3's row h is
//      [P_0 row h | P_1 row h | ...] = head h's [i x j] block in row-major order, 2 KB contiguous.
//   3. Each 2 KB page of c_3 tilized alone (block width 1: rows of 64 B) -> the output tile of head h.
//   4. * SCALE (SFPU on the fp32 dest), + the mask tile (dest reused as srcA), packed bf16 for the writer.
// The packer does the re-tiling that a tile-crossing permute or a face-row NOC gather would otherwise do.
#include <cstdint>
#define REDUCE_OP PoolType::SUM          // tilize.h guards tilize_block on these
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

constexpr uint32_t KT = get_compile_time_arg_val(0);
constexpr uint32_t HT = get_compile_time_arg_val(1);
constexpr uint32_t NT = get_compile_time_arg_val(2);
constexpr uint32_t I = get_compile_time_arg_val(3);
constexpr uint32_t SCALE = get_compile_time_arg_val(4);   // fp32 bits
constexpr uint32_t SW = 4;                                // tiles per fp32 half-sync dest
constexpr uint32_t cb_w = 0, cb_z = 1, cb_m = 2, cb_rm = 3, cb_t = 4, cb_out = 16;

void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    mm_init(cb_w, cb_z, cb_rm, 1);
    cb_wait_front(cb_w, HT * KT);
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t ib = (u / NT) % NT;
        const uint32_t nv = I - ib * 32 < 32 ? I - ib * 32 : 32;
        cb_wait_front(cb_z, 32 * KT);
        cb_wait_front(cb_m, 1);
        for (uint32_t hb = 0; hb < HT; ++hb) {
            mm_init_short(cb_w, cb_z, 1);
            pack_untilize_dest_init<SW, 32>(cb_rm);
            cb_reserve_back(cb_rm, 32);
            for (uint32_t g = 0; g < 32 / SW; ++g) {
                tile_regs_acquire();
                for (uint32_t t = 0; t < SW; ++t) {
                    const uint32_t il = g * SW + t;
                    if (il < nv) {
                        for (uint32_t c = 0; c < KT; ++c) {
                            matmul_tiles(cb_w, cb_z, hb * KT + c, il * KT + c, t);
                        }
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<SW, 32>(cb_rm, 1, g);
                tile_regs_release();
            }
            pack_untilize_uninit(cb_rm);
            cb_push_back(cb_rm, 32);

            tilize_init(cb_rm, 1, cb_t);
            for (uint32_t h = 0; h < 32; ++h) {
                cb_wait_front(cb_rm, 1);
                cb_reserve_back(cb_t, 1);
                tilize_block(cb_rm, 1, cb_t);
                cb_push_back(cb_t, 1);
                cb_pop_front(cb_rm, 1);
            }
            tilize_uninit(cb_rm, cb_t);

            for (uint32_t h0 = 0; h0 < 32; h0 += SW) {
                cb_wait_front(cb_t, SW);
                cb_reserve_back(cb_out, SW);
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_t);
                for (uint32_t t = 0; t < SW; ++t) {
                    copy_tile(cb_t, t, t);
                }
                binop_with_scalar_tile_init();
                for (uint32_t t = 0; t < SW; ++t) {
                    mul_unary_tile(t, SCALE);
                }
                binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_m);
                for (uint32_t t = 0; t < SW; ++t) {
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        cb_m, 0, t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < SW; ++t) {
                    pack_tile(t, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, SW);
                cb_pop_front(cb_t, SW);
            }
        }
        cb_pop_front(cb_z, 32 * KT);
        cb_pop_front(cb_m, 1);
    }
}
