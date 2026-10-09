// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail, weights resident: out = [z +] (o * sigmoid(g)) @ Wo, one output row tile
// (Kt in, Nt out) per unit. The gate runs in DST: g is copied in, sigmoid applied, and o multiplied
// on the FPU with the sigmoid moved to srcA, so sigmoid(g) is never packed. The product is packed to
// bf16 (c_4) for the matmul, which takes the whole K in one block, Nt in DW-wide strips. With RESID
// the projection is packed to bf16 (c_5) and z added on the FPU, the order production's add_ uses.
// RNE bit 0 rounds the gated product to nearest-even before its pack (the pack truncates), bit 1
// the output. KB1 takes K one tile per DST pass and sums the passes in the packer (fp32 L1
// accumulate into c_6), the order ttnn's matmul uses at in0_block_w 1: on Wormhole, fp32 DST
// accumulation across a multi-tile K block writes rare elements off by exactly 1, 2 or 4.
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "../trimul_gin_moved/sigmoid_poly.hpp"

ALWI void sigmoid_bf16_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid<false, false>(idst, (int)VectorMode::RC)));
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t RESID = get_compile_time_arg_val(2);
    constexpr uint32_t SIGPOLY = get_compile_time_arg_val(3);
    constexpr uint32_t RNE = get_compile_time_arg_val(4);
    constexpr uint32_t DW = get_compile_time_arg_val(5);
    constexpr uint32_t KB1 = get_compile_time_arg_val(6);
    const uint32_t nunits = get_arg_val<uint32_t>(0);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t o_cb = tt::CBIndex::c_1;
    constexpr uint32_t g_cb = tt::CBIndex::c_2;
    constexpr uint32_t z_cb = tt::CBIndex::c_3;
    constexpr uint32_t x_cb = tt::CBIndex::c_4;
    constexpr uint32_t u_cb = tt::CBIndex::c_5;
    constexpr uint32_t acc_cb = tt::CBIndex::c_6;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t mm_cb = KB1 ? acc_cb : RESID ? u_cb : out_cb;
    constexpr uint32_t sum_cb = KB1 ? acc_cb : u_cb;   // the update the residual add reads

    mm_init(x_cb, w_cb, KB1 ? x_cb : mm_cb);
    sigmoid_tile_init();
    cb_wait_front(w_cb, Kt * Nt);

    for (uint32_t u = 0; u < nunits; ++u) {
        // gate: x = o * sigmoid(g)
        cb_wait_front(g_cb, Kt);
        cb_wait_front(o_cb, Kt);
        cb_reserve_back(x_cb, Kt);
        if constexpr (KB1) {
            reconfig_data_format(g_cb, o_cb);
            pack_reconfig_data_format(x_cb);
        }
        for (uint32_t k0 = 0; k0 < Kt; k0 += DW) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(g_cb);
            for (uint32_t i = 0; i < DW; ++i) copy_tile(g_cb, k0 + i, i);
            for (uint32_t i = 0; i < DW; ++i) {
                if constexpr (SIGPOLY) sigmoid_poly_tile(i);
                else sigmoid_bf16_tile(i);
            }
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(o_cb);
            for (uint32_t i = 0; i < DW; ++i)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    o_cb, k0 + i, i);
            if constexpr (RNE & 1) {
                for (uint32_t i = 0; i < DW; ++i) round_bf16_rne_tile(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < DW; ++i) pack_tile(i, x_cb);
            tile_regs_release();
        }
        cb_push_back(x_cb, Kt);
        cb_pop_front(g_cb, Kt);
        cb_pop_front(o_cb, Kt);

        // projection: x @ Wo, all of K in one block
        cb_wait_front(x_cb, Kt);
        cb_reserve_back(mm_cb, Nt);
        mm_block_init_short(x_cb, w_cb, false, DW, 1, Kt);
        if constexpr (KB1) {
            pack_reconfig_data_format(acc_cb);
            for (uint32_t n0 = 0; n0 < Nt; n0 += DW) {
                for (uint32_t k = 0; k < Kt; ++k) {
                    tile_regs_acquire();
                    matmul_block(x_cb, w_cb, k, k * Nt + n0, 0, false, DW, 1, Kt);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_l1_acc(k > 0);
                    for (uint32_t i = 0; i < DW; ++i) pack_tile<true>(i, acc_cb, n0 + i);
                    tile_regs_release();
                }
            }
            pack_reconfig_l1_acc(0);
            pack_reconfig_data_format(out_cb);
        } else
        for (uint32_t n0 = 0; n0 < Nt; n0 += DW) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; ++k) matmul_block(x_cb, w_cb, k, k * Nt + n0, 0, false, DW, 1, Kt);
            if constexpr (!RESID && (RNE & 2)) {
                for (uint32_t i = 0; i < DW; ++i) round_bf16_rne_tile(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < DW; ++i) pack_tile(i, mm_cb);
            tile_regs_release();
        }
        cb_push_back(mm_cb, Nt);
        cb_pop_front(x_cb, Kt);

        if constexpr (KB1 && !RESID) {   // the fp32 sum to the bf16 output
            cb_wait_front(acc_cb, Nt);
            cb_reserve_back(out_cb, Nt);
            copy_tile_to_dst_init_short_with_dt(x_cb, acc_cb);
            for (uint32_t n0 = 0; n0 < Nt; n0 += DW) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < DW; ++i) copy_tile(acc_cb, n0 + i, i);
                if constexpr (RNE & 2) {
                    for (uint32_t i = 0; i < DW; ++i) round_bf16_rne_tile(i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < DW; ++i) pack_tile(i, out_cb);
                tile_regs_release();
            }
            cb_push_back(out_cb, Nt);
            cb_pop_front(acc_cb, Nt);
        }

        if constexpr (RESID) {
            cb_wait_front(sum_cb, Nt);
            cb_wait_front(z_cb, Nt);
            cb_reserve_back(out_cb, Nt);
            if constexpr (KB1) reconfig_data_format(z_cb, acc_cb);
            add_tiles_init(z_cb, sum_cb);
            for (uint32_t n0 = 0; n0 < Nt; n0 += DW) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < DW; ++i) add_tiles(z_cb, sum_cb, n0 + i, n0 + i, i);
                if constexpr (RNE & 2) {
                    for (uint32_t i = 0; i < DW; ++i) round_bf16_rne_tile(i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < DW; ++i) pack_tile(i, out_cb);
                tile_regs_release();
            }
            cb_push_back(out_cb, Nt);
            cb_pop_front(sum_cb, Nt);
            cb_pop_front(z_cb, Nt);
        }
    }
}
