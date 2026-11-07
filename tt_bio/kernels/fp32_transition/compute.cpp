// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fp32_transition compute (tt_bio/fp32_transition.py): `tenstorrent._fp32_single_transition` for one 32-token row tile,
// as one program. Same algebra and rounding points as the composed form:
//   xn = LN(x) in fp32 (fp32_ln's SFPU steps), c0 = token 0's xn row, d = bf16(xn - c0), c0b = bf16(c0)
//   a = d @ w1 + c0b @ w1, b = d @ w2 + c0b @ w2 (fp32 dest), h = silu(a) * b (SFPU, fp32)
//   h0 = token 0's h = silu(c0b @ w1) * (c0b @ w2), since token 0's d is zero
//   out = bf16(h - h0) @ w3 + bf16(h0) @ w3, one fp32 dest sum, rounded once to the output format
// The reader hands this core token 0's x row copied over all 32 rows of a tile (c_8), so c0, c0b @ w and h0 come out
// already broadcast over the rows and no core needs another's results. Only the fp32 summation order differs from the
// composed form. Each matmul keeps all of its K in dest, as ttnn.linear does with the dest guard off (dest_guard.py).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "../trimul_gin_moved/sigmoid_poly.hpp"

constexpr uint32_t KT = get_compile_time_arg_val(0);       // channel tiles (384 -> 12)
constexpr uint32_t NH = get_compile_time_arg_val(1);       // hidden tiles (1536 -> 48)
constexpr uint32_t INV_K = get_compile_time_arg_val(2);    // fp32 bits of 1 / channels
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(3);
constexpr uint32_t OUT_BF16 = get_compile_time_arg_val(4);
constexpr uint32_t KC3 = get_compile_time_arg_val(5);      // fc3 K tiles a w3 chunk

constexpr uint32_t cb_x = 0, cb_g = 1, cb_b = 2, cb_xc = 3, cb_s1 = 4, cb_m = 5, cb_s2 = 6, cb_r = 7, cb_x0 = 8;
constexpr uint32_t cb_d = 9, cb_c0b = 10, cb_w1 = 11, cb_w2 = 12, cb_w3 = 13, cb_dh = 14, cb_h0b = 15;
constexpr uint32_t cb_out = 16, cb_c0 = 17;

// The matmuls leave unpacker A on bf16, and every LN load is fp32: reconfigure it on each copy.
ALWI void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, tile, dst);
}

ALWI void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(0, cb);
    tile_regs_release();
}

// DEST 0 <- row sums of the KT tiles of `cb` (squared first when SQ), times 1 / K, in column 0 (fp32_ln's row_mean).
template <bool SQ>
ALWI void row_mean(uint32_t cb) {
    load(cb, 0, 0);
    if constexpr (SQ) {
        square_tile_init();
        square_tile(0);
    }
    for (uint32_t k = 1; k < KT; ++k) {
        load(cb, k, 1);
        if constexpr (SQ) {
            square_tile_init();
            square_tile(1);
        }
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_K);
}

// LN of the KT fp32 tiles in `src` (the reader broadcasts the two statistics back through c_5 / c_7). TOKEN0: the tile
// is token 0's row broadcast, so pack each normed tile fp32 to c_17 (c0) and rounded to c_10 (c0b); else pack
// bf16(xn - c0) to c_9 (d).
template <bool TOKEN0>
void layer_norm(uint32_t src) {
    cb_wait_front(src, KT);
    cb_reserve_back(cb_s1, 1);
    tile_regs_acquire();
    row_mean<false>(src);
    pack_one(cb_s1);
    cb_push_back(cb_s1, 1);

    cb_wait_front(cb_m, 1);
    cb_reserve_back(cb_xc, KT);
    for (uint32_t k = 0; k < KT; ++k) {
        tile_regs_acquire();
        load(src, k, 0);
        load(cb_m, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        pack_one(cb_xc);
    }
    cb_push_back(cb_xc, KT);
    cb_pop_front(src, KT);
    cb_pop_front(cb_m, 1);

    cb_wait_front(cb_xc, KT);
    cb_reserve_back(cb_s2, 1);
    tile_regs_acquire();
    row_mean<true>(cb_xc);
    add_unary_tile(0, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    pack_one(cb_s2);
    cb_push_back(cb_s2, 1);

    cb_wait_front(cb_r, 1);
    if constexpr (TOKEN0) {
        cb_reserve_back(cb_c0, KT);
        cb_reserve_back(cb_c0b, KT);
    } else {
        cb_wait_front(cb_c0, KT);
        cb_reserve_back(cb_d, KT);
    }
    for (uint32_t k = 0; k < KT; ++k) {
        tile_regs_acquire();
        load(cb_xc, k, 0);
        load(cb_r, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        load(cb_g, k, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        load(cb_b, k, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        if constexpr (TOKEN0) {
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_c0);
            pack_tile(0, cb_c0, k);
            tile_regs_release();
            tile_regs_acquire();
            load(cb_c0, k, 0);   // the packed fp32 copy, so c0b rounds exactly the c0 every d subtracts
            round_bf16_rne_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_c0b);
            pack_tile(0, cb_c0b, k);
            tile_regs_release();
        } else {
            load(cb_c0, k, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            round_bf16_rne_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_d);
            pack_tile(0, cb_d, k);
            tile_regs_release();
        }
    }
    if constexpr (TOKEN0) {
        cb_push_back(cb_c0, KT);
        cb_push_back(cb_c0b, KT);
    } else {
        cb_push_back(cb_d, KT);
    }
    cb_pop_front(cb_xc, KT);
    cb_pop_front(cb_r, 1);
}

// One row tile already normed into c_9 (d): fc1 / fc2 per hidden tile n, dest 0 = d @ w1, 1 = c0b @ w1, 2 = d @ w2,
// 3 = c0b @ w2 (the reader streams column n of w1 and of w2, KT tiles each); h and h0 leave as bf16(h - h0) and bf16(h0).
// Then fc3 per output tile m: both K runs into dest 0 over the same w3 column (KC3 tiles a chunk), rounded once.
void transition_row() {
    cb_wait_front(cb_d, KT);
    cb_reserve_back(cb_dh, NH);
    cb_reserve_back(cb_h0b, NH);
    for (uint32_t n = 0; n < NH; ++n) {
        cb_wait_front(cb_w1, KT);
        cb_wait_front(cb_w2, KT);
        tile_regs_acquire();
        mm_init_short(cb_d, cb_w1, 0);
        reconfig_data_format(cb_w1, cb_d);
        for (uint32_t k = 0; k < KT; ++k) {
            matmul_tiles(cb_d, cb_w1, k, k, 0);
            matmul_tiles(cb_c0b, cb_w1, k, k, 1);
            matmul_tiles(cb_d, cb_w2, k, k, 2);
            matmul_tiles(cb_c0b, cb_w2, k, k, 3);
        }
        cb_pop_front(cb_w1, KT);
        cb_pop_front(cb_w2, KT);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);     // a
        add_binary_tile(2, 3, 2);     // b
        silu_tile_init();
        silu_tile(0);
        silu_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);     // h
        mul_binary_tile(1, 3, 1);     // h0
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);     // h - h0
        round_bf16_rne_tile(0);
        round_bf16_rne_tile(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_dh);
        pack_tile(0, cb_dh, n);
        pack_reconfig_data_format(cb_h0b);
        pack_tile(1, cb_h0b, n);
        tile_regs_release();
    }
    cb_push_back(cb_dh, NH);
    cb_push_back(cb_h0b, NH);
    cb_pop_front(cb_d, KT);
    cb_wait_front(cb_dh, NH);
    cb_wait_front(cb_h0b, NH);
    for (uint32_t m = 0; m < KT; ++m) {
        tile_regs_acquire();
        mm_init_short(cb_dh, cb_w3, 0);
        reconfig_data_format(cb_w3, cb_dh);
        for (uint32_t k0 = 0; k0 < NH; k0 += KC3) {
            cb_wait_front(cb_w3, KC3);
            for (uint32_t k = 0; k < KC3; ++k) {
                matmul_tiles(cb_dh, cb_w3, k0 + k, k, 0);
                matmul_tiles(cb_h0b, cb_w3, k0 + k, k, 0);
            }
            cb_pop_front(cb_w3, KC3);
        }
        if constexpr (OUT_BF16) round_bf16_rne_tile(0);
        cb_reserve_back(cb_out, 1);
        pack_one(cb_out);
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_dh, NH);
    cb_pop_front(cb_h0b, NH);
}

void kernel_main() {
    const uint32_t nr = get_arg_val<uint32_t>(0);
    binary_op_init_common(cb_x, cb_x, cb_out);
    mm_init(cb_d, cb_w1, cb_dh, 0);
    cb_wait_front(cb_g, KT);
    cb_wait_front(cb_b, KT);
    layer_norm<true>(cb_x0);
    cb_wait_front(cb_c0b, KT);
    for (uint32_t r = 0; r < nr; ++r) {
        layer_norm<false>(cb_x);
        transition_row();
    }
    cb_pop_front(cb_c0b, KT);
    cb_pop_front(cb_c0, KT);
    cb_pop_front(cb_g, KT);
    cb_pop_front(cb_b, KT);
}
