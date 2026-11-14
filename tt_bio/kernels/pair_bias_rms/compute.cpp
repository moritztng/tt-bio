// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_bias_rms: attention pair biases straight from the pair tensor, RMSNorm included, head-first,
//   out[b, h, i, j] = (z[b, i, j, :] @ W[:, h]) * rsqrt(mean_c z[b, i, j, c]^2 + eps),
// for 32 heads (one head tile). dit_pair_bias's layout trick (kernels/dit_pair_bias/compute.cpp) with the norm
// folded in: the norm is a per-(i, j) scale, so it can follow the projection. Per unit (b, ib, jb), two rows i
// at a time:
//   1. sq = z(i, jb)^2, elementwise, packed bf16.
//   2. P_i = W^T @ z(i, jb)^T into dest t, S_i = ONES @ sq^T into dest t + 2. Tile P_i is [h x j]; every row of
//      S_i holds sum_c z[i, j, c]^2 for its column j, so it lines up with P_i elementwise.
//   3. P_i *= rsqrt(S_i / C + eps) on the SFPU in the fp32 dest, then the pair packed untilized into one 32-tile
//      row of c_3 as in dit_pair_bias, and each 2 KB row tilized into head h's output tile.
// z streams through c_1 two rows at a time; nothing of a unit stays resident but the 32 untilized rows.
// PB_Z8 (`pair_bfp8z`): z is bfloat8_b, so srcA / srcB are reconfigured at each step that changes operands.
#include <cstdint>
#define REDUCE_OP PoolType::SUM          // tilize.h guards tilize_block on these
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#ifdef PB_Z8
#include "api/compute/reconfig_data_format.h"
#endif
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t KT = get_compile_time_arg_val(0);
constexpr uint32_t INV_C = get_compile_time_arg_val(1);   // fp32 bits of 1 / C
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(2);  // fp32 bits of eps (EPS is an LLK macro)
constexpr uint32_t ROWS = 2;  // rows i per dest acquire: P in 0..1, S in 2..3
constexpr uint32_t cb_w = 0, cb_z = 1, cb_one = 2, cb_rm = 3, cb_sq = 4, cb_out = 16;

void kernel_main() {
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    mm_init(cb_w, cb_z, cb_rm, 1);
    cb_wait_front(cb_w, KT);
    cb_wait_front(cb_one, 1);
    for (uint32_t u = 0; u < nunits; ++u) {
        cb_reserve_back(cb_rm, 32);
        for (uint32_t g = 0; g < 32 / ROWS; ++g) {
            cb_wait_front(cb_z, ROWS * KT);
            // 1. squares
#ifdef PB_Z8
            reconfig_data_format(cb_z, cb_z);
#endif
            mul_tiles_init(cb_z, cb_z);
            cb_reserve_back(cb_sq, ROWS * KT);
            for (uint32_t t0 = 0; t0 < ROWS * KT; t0 += 4) {
                const uint32_t m = ROWS * KT - t0 < 4 ? ROWS * KT - t0 : 4;
                tile_regs_acquire();
                for (uint32_t k = 0; k < m; ++k) {
                    mul_tiles(cb_z, cb_z, t0 + k, t0 + k, k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < m; ++k) {
                    pack_tile(k, cb_sq);
                }
                tile_regs_release();
            }
            cb_push_back(cb_sq, ROWS * KT);
            cb_wait_front(cb_sq, ROWS * KT);
            // 2. projection and sum of squares, 3. the norm's scale
            tile_regs_acquire();
#ifdef PB_Z8
            reconfig_data_format(cb_z, cb_w);   // srcA takes in1 (z), srcB in0 (W^T)
#endif
            mm_init_short(cb_w, cb_z, 1);
            for (uint32_t t = 0; t < ROWS; ++t) {
                for (uint32_t c = 0; c < KT; ++c) {
                    matmul_tiles(cb_w, cb_z, c, t * KT + c, t);
                }
            }
#ifdef PB_Z8
            reconfig_data_format(cb_sq, cb_one);
#endif
            mm_init_short(cb_one, cb_sq, 1);
            for (uint32_t t = 0; t < ROWS; ++t) {
                for (uint32_t c = 0; c < KT; ++c) {
                    matmul_tiles(cb_one, cb_sq, 0, t * KT + c, ROWS + t);
                }
            }
            binop_with_scalar_tile_init();
            for (uint32_t t = 0; t < ROWS; ++t) {
                mul_unary_tile(ROWS + t, INV_C);
                add_unary_tile(ROWS + t, EPS_BITS);
            }
            rsqrt_tile_init();
            for (uint32_t t = 0; t < ROWS; ++t) {
                rsqrt_tile(ROWS + t);
            }
            mul_binary_tile_init();
            for (uint32_t t = 0; t < ROWS; ++t) {
                mul_binary_tile(t, ROWS + t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest_init<ROWS, 32>(cb_rm);
            pack_untilize_dest<ROWS, 32>(cb_rm, 1, g);
            pack_untilize_uninit(cb_rm);
            tile_regs_release();
            cb_pop_front(cb_sq, ROWS * KT);
            cb_pop_front(cb_z, ROWS * KT);
        }
        cb_push_back(cb_rm, 32);
        tilize_init(cb_rm, 1, cb_out);
        for (uint32_t h = 0; h < 32; ++h) {
            cb_wait_front(cb_rm, 1);
            cb_reserve_back(cb_out, 1);
            tilize_block(cb_rm, 1, cb_out);
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_rm, 1);
        }
        tilize_uninit(cb_rm, cb_out);
    }
}
