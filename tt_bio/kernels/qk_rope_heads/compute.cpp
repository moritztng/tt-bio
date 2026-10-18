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

// LayerNorm(x) * gamma, no beta, Kt tiles of C::x (popped) into C::xc (pushed), fp32.
template <class C>
ALWI void ln_no_beta() {
    using namespace ln_prologue;
    row_mean<C>(C::x, C::mean, false);
    cb_wait_front(C::mean, 1);
    stage<C, SUB_COL>(C::x, C::mean, C::xc);
    cb_pop_front(C::x, C::Kt);
    cb_pop_front(C::mean, 1);
    cb_wait_front(C::xc, C::Kt);
    stage<C, MUL>(C::xc, C::xc, C::tmp);
    row_mean<C>(C::tmp, C::var, true);
    rstd<C>();
    stage<C, MUL_COL>(C::xc, C::rstd, C::tmp);
    cb_pop_front(C::xc, C::Kt);
    cb_pop_front(C::rstd, 1);
    cb_wait_front(C::tmp, C::Kt);
    stage<C, MUL_ROW>(C::tmp, C::gamma, C::xc);
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
#ifndef QKRH_RELOAD
    cb_wait_front(cb_gq, KT);
    cb_wait_front(cb_gk, KT);
#endif
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    for (uint32_t u = first; u < first + count; ++u) {   // the q / k units; v is the reader's and the writer's
        [[maybe_unused]] const uint32_t t = u / R;   // 0 q, 1 k
#ifdef QKRH_RELOAD
        cb_wait_front(cb_gq, KT);
        ln_no_beta<Ln<cb_gq>>();
        cb_pop_front(cb_gq, KT);
#else
        if (t == 0) {
            ln_no_beta<Ln<cb_gq>>();
        } else {
            ln_no_beta<Ln<cb_gk>>();
        }
#endif
        cb_wait_front(cb_n, KT);
        cb_wait_front(cb_cos, 1);
        cb_wait_front(cb_sin, 1);
        for (uint32_t h = 0; h < KT / 2; ++h) {
            const uint32_t a = 2 * h, b = a + 1;
            reconfig_data_format(cb_n, cb_cos);
            pack_reconfig_data_format(cb_p);
            mul_tiles_init(cb_n, cb_cos);
            cb_reserve_back(cb_p, 2);
            tile_regs_acquire();
            mul_tiles(cb_n, cb_cos, a, 0, 0);
            mul_tiles(cb_n, cb_cos, b, 0, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_p);
            pack_tile(1, cb_p);
            tile_regs_release();
            cb_push_back(cb_p, 2);

            cb_wait_front(cb_p, 2);
            cb_reserve_back(cb_out, 2);
            reconfig_data_format(cb_n, cb_sin);
            pack_reconfig_data_format(cb_out);
            mul_tiles_init(cb_n, cb_sin);
            tile_regs_acquire();
            mul_tiles(cb_n, cb_sin, b, 0, 0);
            mul_tiles(cb_n, cb_sin, a, 0, 1);
            reconfig_data_format_srca(cb_p);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_p);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_p, 0, 0);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_p);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_p, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            pack_tile(1, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 2);
            cb_pop_front(cb_p, 2);
        }
        cb_pop_front(cb_n, KT);
        cb_pop_front(cb_cos, 1);
        cb_pop_front(cb_sin, 1);
    }
}
