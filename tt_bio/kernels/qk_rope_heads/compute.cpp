// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qk_rope_heads compute: for a q or k row tile (KT tiles in c_5), LayerNorm over D with gamma and no beta (ESMC's
// q_ln / k_ln), left in fp32 in c_17 (the centred-x CB, free by then), then NeoX RoPE per head on those tiles: head h is tiles a = 2h, b = 2h + 1 and
// rotate_half swaps them with a sign, so
//     out_a = x_a * cos - x_b * sin        out_b = x_b * cos + x_a * sin
// with one cos / sin tile for the row (c_6 / c_7). The cos products go through an fp32 CB (c_20); the sin products
// stay in dest and the add / subtract reuses dest as srcB. Two output tiles per head into c_16. v units are the
// reader's and the writer's alone.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "../common/ln_prologue.hpp"

constexpr uint32_t KT = get_compile_time_arg_val(0);

template <uint32_t G>
struct Ln {
    static constexpr uint32_t Kt = KT, x = 5, gamma = G, beta = G, scaler = 10, eps = 11, mean = 12, var = 13,
                              rstd = 14, xc = 17, tmp = 18;
};

// Tiles per dest acquire in the norm stages (QKRH_BLK, 1 or 4; 4 fits the half-sync fp32 dest) and q / k heads per
// RoPE group (QKRH_HG, 1 or 2). The host sets HG = 2 only where it divides KT / 2: a group's c_20 / c_16 reservation
// must never straddle the end of its CB, which an odd head count (300M, 15) would do on its last group.
#ifndef QKRH_BLK
#define QKRH_BLK 1
#endif
#ifndef QKRH_HG
#define QKRH_HG 1
#endif
constexpr uint32_t BLK = QKRH_BLK;
constexpr uint32_t HG = QKRH_HG;

// LayerNorm(x) * gamma, no beta (skipped under QKRH_NO_LN, where the RoPE reads c_5 directly), Kt tiles of C::x (popped) into C::xc (pushed), fp32.
template <class C>
ALWI void ln_no_beta() {
    using namespace ln_prologue;
    row_mean<C>(C::x, C::mean, false);
    cb_wait_front(C::mean, 1);
    stage<C, SUB_COL, BLK>(C::x, C::mean, C::xc);
    cb_pop_front(C::x, C::Kt);
    cb_pop_front(C::mean, 1);
    cb_wait_front(C::xc, C::Kt);
    stage<C, MUL, BLK>(C::xc, C::xc, C::tmp);
    row_mean<C>(C::tmp, C::var, true);
    rstd<C>();
    cb_wait_front(C::xc, C::Kt);
    stage<C, MUL_COL, BLK>(C::xc, C::rstd, C::tmp);
    cb_pop_front(C::xc, C::Kt);
    cb_pop_front(C::rstd, 1);
    cb_wait_front(C::tmp, C::Kt);
    stage<C, MUL_ROW, BLK>(C::tmp, C::gamma, C::xc);
    cb_pop_front(C::tmp, C::Kt);
}

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(1);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_x = 5, cb_cos = 6, cb_sin = 7, cb_gq = 8, cb_scaler = 10, cb_eps = 11, cb_out = 16,
                       cb_n = 17, cb_p = 20;
    [[maybe_unused]] constexpr uint32_t cb_gk = 15;
    if (count == 0) {
        return;
    }
    binary_op_init_common(cb_x, cb_gq, cb_n);
#if !defined(QKRH_RELOAD) && !defined(QKRH_NO_LN)
    cb_wait_front(cb_gq, KT);
    cb_wait_front(cb_gk, KT);
#endif
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    for (uint32_t u = first; u < first + count; ++u) {   // the q / k units; v is the reader's and the writer's
        [[maybe_unused]] const uint32_t t = u / R;   // 0 q, 1 k
#if defined(QKRH_NO_LN)
        constexpr uint32_t src = cb_x;   // RoPE straight from the input row
#elif defined(QKRH_RELOAD)
        constexpr uint32_t src = cb_n;
        cb_wait_front(cb_gq, KT);
        ln_no_beta<Ln<cb_gq>>();
        cb_pop_front(cb_gq, KT);
#else
        constexpr uint32_t src = cb_n;
        if (t == 0) {
            ln_no_beta<Ln<cb_gq>>();
        } else {
            ln_no_beta<Ln<cb_gk>>();
        }
#endif
        cb_wait_front(src, KT);
        cb_wait_front(cb_cos, 1);
        cb_wait_front(cb_sin, 1);
        for (uint32_t h0 = 0; h0 < KT / 2; h0 += HG) {   // HG heads (2 HG tiles) a dest acquire
            const uint32_t n = KT / 2 - h0 < HG ? KT / 2 - h0 : HG, a0 = 2 * h0;
            reconfig_data_format(src, cb_cos);
            pack_reconfig_data_format(cb_p);
            mul_tiles_init(src, cb_cos);
            cb_reserve_back(cb_p, 2 * n);
            tile_regs_acquire();
            for (uint32_t j = 0; j < 2 * n; ++j) mul_tiles(src, cb_cos, a0 + j, 0, j);   // a cos, b cos per head
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < 2 * n; ++j) pack_tile(j, cb_p);
            tile_regs_release();
            cb_push_back(cb_p, 2 * n);

            // dest j < n: b_j sin (for out a_j), dest n + j: a_j sin (for out b_j); packed a_0, b_0, a_1, b_1, ...
            cb_wait_front(cb_p, 2 * n);
            cb_reserve_back(cb_out, 2 * n);
            reconfig_data_format(src, cb_sin);
            pack_reconfig_data_format(cb_out);
            mul_tiles_init(src, cb_sin);
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                mul_tiles(src, cb_sin, a0 + 2 * j + 1, 0, j);
                mul_tiles(src, cb_sin, a0 + 2 * j, 0, n + j);
            }
            reconfig_data_format_srca(cb_p);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_p);
            for (uint32_t j = 0; j < n; ++j) {
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_p, 2 * j, j);
            }
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_p);
            for (uint32_t j = 0; j < n; ++j) {
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_p, 2 * j + 1, n + j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_out);
                pack_tile(n + j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, 2 * n);
            cb_pop_front(cb_p, 2 * n);
        }
        cb_pop_front(src, KT);
        cb_pop_front(cb_cos, 1);
        cb_pop_front(cb_sin, 1);
    }
}
