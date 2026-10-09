// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail, weights resident: out = [z +] (xa @ wa) * sigmoid(xb @ wb), one Mb x Nt block at a
// time. Both weights sit in c_1 for the whole program ([wa | wb], each Kt x Nt in row-major tile
// order) and are never popped. The epilogue is trimul_tail's EPI 1 / 2 op for op (one K block
// accumulated in DST in k order, each pass packed straight to bf16, sigmoid in DST on the gate
// pass, FPU multiply, residual added in DST), so the output is the same bits.
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

ALWI void sigmoid_bf16_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid<false, false>(idst, (int)VectorMode::RC)));
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Mb = get_compile_time_arg_val(2);
    constexpr uint32_t SBW = get_compile_time_arg_val(3);
    constexpr uint32_t SHARED = get_compile_time_arg_val(4);
    constexpr uint32_t RESID = get_compile_time_arg_val(5);
    const uint32_t nblocks = get_arg_val<uint32_t>(0);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t p_cb = tt::CBIndex::c_4;
    constexpr uint32_t g_cb = tt::CBIndex::c_5;
    constexpr uint32_t z_cb = tt::CBIndex::c_7;
    constexpr uint32_t blk = Mb * Nt;

    sigmoid_tile_init();
    mm_init(in0_cb, in1_cb, p_cb);
    cb_wait_front(in1_cb, 2 * Kt * Nt);

    for (uint32_t b = 0; b < nblocks; ++b) {
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const uint32_t pass_cb = pass == 0 ? p_cb : g_cb;
            mm_block_init_short(in0_cb, in1_cb, false, SBW, 1, Kt);
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(pass_cb);
            cb_wait_front(in0_cb, Mb * Kt);
            cb_reserve_back(pass_cb, blk);
            for (uint32_t mi = 0; mi < Mb; ++mi) {
                for (uint32_t n0 = 0; n0 < Nt; n0 += SBW) {
                    tile_regs_acquire();
                    uint32_t i0 = mi * Kt;
                    uint32_t i1 = pass * Kt * Nt + n0;
                    for (uint32_t k = 0; k < Kt; ++k) {
                        matmul_block(in0_cb, in1_cb, i0, i1, 0, false, SBW, 1, Kt);
                        i0++;
                        i1 += Nt;
                    }
                    if (pass == 1) {
                        for (uint32_t i = 0; i < SBW; ++i) sigmoid_bf16_tile(i);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < SBW; ++i) pack_tile<true>(i, pass_cb, mi * Nt + n0 + i);
                    tile_regs_release();
                }
            }
            cb_push_back(pass_cb, blk);
            if (!(SHARED && pass == 0)) cb_pop_front(in0_cb, Mb * Kt);
        }

        cb_wait_front(p_cb, blk);
        cb_wait_front(g_cb, blk);
        if constexpr (RESID) cb_wait_front(z_cb, blk);
        cb_reserve_back(out_cb, blk);
        reconfig_data_format(p_cb, g_cb);
        pack_reconfig_data_format(out_cb);
        for (uint32_t t0 = 0; t0 < blk; t0 += 4) {
            tile_regs_acquire();
            mul_tiles_init(p_cb, g_cb);
            for (uint32_t i = 0; i < 4; ++i) mul_tiles(p_cb, g_cb, t0 + i, t0 + i, i);
            if constexpr (RESID) {
                binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    z_cb);
                for (uint32_t i = 0; i < 4; ++i)
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        z_cb, t0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < 4; ++i) pack_tile(i, out_cb);
            tile_regs_release();
        }
        cb_push_back(out_cb, blk);
        cb_pop_front(p_cb, blk);
        cb_pop_front(g_cb, blk);
        if constexpr (RESID) cb_pop_front(z_cb, blk);
    }
}
