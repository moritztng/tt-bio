// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fp32_ln compute (tt_bio/fp32_ln.py): the layer norm of this core's row tiles of an fp32 x, every step on the SFPU
// in an fp32 DEST. The inputs are unpacked straight to DEST (UnpackToDestFp32), never through srcA, whose tf32 width
// loses a single track's per-token part. Per row tile: the KT tiles folded by SFPU adds and their 32 columns summed by
// sfpu_reduce (REDUCE_ROW, sums in column 0) give the mean, which goes to the reader and comes back broadcast over the
// columns (c_5); x - mean to c_3; the same for the biased variance of x - mean, whose rsqrt comes back as c_7; then
// (x - mean) * rstd * gamma + beta, gamma and beta full fp32 tiles, packed to c_16 (rounded to nearest even for bf16).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/tile_move_copy.h"
#include "../trimul_gin_moved/sigmoid_poly.hpp"

constexpr uint32_t KT = get_compile_time_arg_val(0);
constexpr uint32_t INV_K = get_compile_time_arg_val(1);   // fp32 bits of 1 / K
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(2);     // fp32 bits
constexpr uint32_t OUT_BF16 = get_compile_time_arg_val(3);

constexpr uint32_t cb_x = tt::CBIndex::c_0, cb_g = tt::CBIndex::c_1, cb_b = tt::CBIndex::c_2;
constexpr uint32_t cb_xc = tt::CBIndex::c_3, cb_s1 = tt::CBIndex::c_4, cb_m = tt::CBIndex::c_5;
constexpr uint32_t cb_s2 = tt::CBIndex::c_6, cb_r = tt::CBIndex::c_7, cb_out = tt::CBIndex::c_16;

ALWI void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, tile, dst);
}

// DEST 0 <- row sums of the KT tiles of `cb` (squared first when SQ), times 1 / K, in column 0.
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

ALWI void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(0, cb);
    tile_regs_release();
}

void kernel_main() {
    const uint32_t nr = get_arg_val<uint32_t>(0);
    binary_op_init_common(cb_x, cb_x, cb_out);
    cb_wait_front(cb_g, KT);
    cb_wait_front(cb_b, KT);
    for (uint32_t r = 0; r < nr; ++r) {
        cb_wait_front(cb_x, KT);
        cb_reserve_back(cb_s1, 1);
        tile_regs_acquire();
        row_mean<false>(cb_x);
        pack_one(cb_s1);
        cb_push_back(cb_s1, 1);

        cb_wait_front(cb_m, 1);
        cb_reserve_back(cb_xc, KT);
        for (uint32_t k = 0; k < KT; ++k) {
            tile_regs_acquire();
            load(cb_x, k, 0);
            load(cb_m, 0, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            pack_one(cb_xc);
        }
        cb_push_back(cb_xc, KT);
        cb_pop_front(cb_x, KT);
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
        for (uint32_t k = 0; k < KT; ++k) {   // one tile at a time, so the writer's stores overlap this stage
            cb_reserve_back(cb_out, 1);
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
            if constexpr (OUT_BF16) round_bf16_rne_tile(0);
            pack_one(cb_out);
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_xc, KT);
        cb_pop_front(cb_r, 1);
    }
    cb_pop_front(cb_g, KT);
    cb_pop_front(cb_b, KT);
}
