// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul gated in-projection with the channel move fused in: per unit (b, xt, yt) and output channel
// tile q of [a | b], the 32 tiles P_il = W^T[q] @ X_il^T (in1 transposed in the unpacker), one per
// il, each [32 channels x 32 y]. Pass 0 is the value, pass 1 the gate (sigmoid in DST), then
// a = p * sigmoid(g) [* m] is packed in il order for the writer, whose gather turns row c of the 32
// tiles into the output tile of channel c. Same epilogue as trimul_tail_res (bf16 pack per pass, FPU
// multiply); the mask is the same per-row factor, broadcast along rows here because the tile is
// transposed.
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

ALWI void sigmoid_bf16_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid<false, false>(idst, (int)VectorMode::RC)));
}

#include "sigmoid_poly.hpp"
// GINSPLIT (lever trimul_gin_split): even channel tiles q go to c_21 for reader.cpp's gather, odd ones to c_2 for the
// writer's, so both data-movement cores issue the channel move's face-row reads (trimul_in_t1's T1SPLIT).
#ifndef GINSPLIT
#define GINSPLIT 0
#endif
#if defined(GIN_PSIG) && defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_silu.h"   // the packer-side SFPU init, as trimul_in_t1's T1_PSIG
#endif

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t CT2 = get_compile_time_arg_val(1);
    constexpr uint32_t SBW = get_compile_time_arg_val(2);
    constexpr uint32_t MASK = get_compile_time_arg_val(3);
    // Stage ablation, diagnostic only: bit 0 no sigmoid, bit 1 no gate multiply, bit 4 no matmul.
    constexpr uint32_t ABL = get_compile_time_arg_val(4);
    constexpr uint32_t SIGPOLY = get_compile_time_arg_val(5);   // 1/2: sigmoid_poly.hpp's degree-8/6 polynomial
    // Round to nearest even before a pack (the pack truncates): bit 0 the output, bit 1 p and g.
    constexpr uint32_t RNE = get_compile_time_arg_val(6);
    // 1: value and gate of one subblock side by side in DST (value at 0..SBW-1, gate at SBW..2SBW-1), the gate's
    // sigmoid and multiply done there and only a = p * sigmoid(g) packed (lever trimul_gin_fuse; unmasked calls).
    // p and g never round-trip through L1 as bf16, which also frees their two 32-tile buffers.
    constexpr uint32_t FUSE = get_compile_time_arg_val(7);
    static_assert(!(FUSE && MASK), "the fused gate has no mask path");
    static_assert(!(FUSE && GINSPLIT), "the split runs on the unfused gate only");
    const uint32_t nunits = get_arg_val<uint32_t>(0);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t x_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t p_cb = tt::CBIndex::c_4;
    constexpr uint32_t g_cb = tt::CBIndex::c_5;
    constexpr uint32_t m_cb = tt::CBIndex::c_6;
    constexpr uint32_t out_rd_cb = tt::CBIndex::c_21;   // GINSPLIT: the even channel tiles, to the reader

    sigmoid_tile_init();
    mm_init(w_cb, x_cb, FUSE ? out_cb : p_cb, 1);
#ifdef GIN_PSIG
    PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));   // the packer thread's SFPU address modes
#endif
    cb_wait_front(w_cb, 2 * CT2 * Kt);

    if constexpr (FUSE) {
        mm_block_init_short(w_cb, x_cb, true, SBW, 1, Kt);
        for (uint32_t u = 0; u < nunits; ++u) {
            cb_wait_front(x_cb, 32 * Kt);
            for (uint32_t q = 0; q < CT2; ++q) {
                cb_reserve_back(out_cb, 32);
                for (uint32_t il0 = 0; il0 < 32; il0 += SBW) {
                    tile_regs_acquire();
                    for (uint32_t pass = 0; pass < 2; ++pass) {
                        uint32_t i0 = (pass * CT2 + q) * Kt;
                        uint32_t i1 = il0;
                        for (uint32_t k = 0; k < Kt; ++k, ++i0, i1 += 32) {
                            matmul_block(w_cb, x_cb, i0, i1, pass * SBW, true, SBW, 1, Kt);
                        }
                    }
                    for (uint32_t i = 0; i < SBW; ++i) gate_poly_tile<SIGPOLY, SBW>(i);
                    if constexpr (RNE & 1) {
                        for (uint32_t i = 0; i < SBW; ++i) round_bf16_rne_tile(i);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < SBW; ++i) pack_tile<true>(i, out_cb, il0 + i);
                    tile_regs_release();
                }
                cb_push_back(out_cb, 32);
            }
            cb_pop_front(x_cb, 32 * Kt);
        }
        return;
    }

    for (uint32_t u = 0; u < nunits; ++u) {
        cb_wait_front(x_cb, 32 * Kt);
        if constexpr (MASK) cb_wait_front(m_cb, 32);
        for (uint32_t q = 0; q < CT2; ++q) {
            for (uint32_t pass = 0; pass < 2; ++pass) {
                const uint32_t pass_cb = pass == 0 ? p_cb : g_cb;
                mm_block_init_short(w_cb, x_cb, true, SBW, 1, Kt);
                reconfig_data_format(x_cb, w_cb);
                pack_reconfig_data_format(pass_cb);
                cb_reserve_back(pass_cb, 32);
                for (uint32_t il0 = 0; il0 < 32; il0 += SBW) {
                    tile_regs_acquire();
                    uint32_t i0 = (pass * CT2 + q) * Kt;
                    uint32_t i1 = il0;
                    for (uint32_t k = 0; k < Kt && !(ABL & 16); ++k, ++i0, i1 += 32) {
                        matmul_block(w_cb, x_cb, i0, i1, 0, true, SBW, 1, Kt);
                    }
#ifdef GIN_PSIG
                    if (false) {
#else
                    if (pass == 1 && !(ABL & 1)) {
#endif
                        for (uint32_t i = 0; i < SBW; ++i) {
                            if constexpr (SIGPOLY) sigmoid_poly_tile<SIGPOLY>(i);
                            else sigmoid_bf16_tile(i);
                        }
                    }
                    if constexpr (RNE & 2) {
                        for (uint32_t i = 0; i < SBW; ++i) round_bf16_rne_tile(i);
                    }
                    tile_regs_commit();
#ifdef GIN_PSIG
                    if (pass == 1) {
                        // GIN_PSIG (lever trimul_gin_psig): the gate sigmoid on the packer thread's SFPU, on the dest
                        // half it holds, so the FPU's next matmul block overlaps it (trimul_in_t1's T1_PSIG pattern).
                        PACK(TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_CFG,
                                         semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
                        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                        if constexpr (!(ABL & 1)) {
                            for (uint32_t i = 0; i < SBW; ++i) sigmoid_poly_tile_pack<SIGPOLY>(i);
                        }
                        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    } else {
                        tile_regs_wait();
                    }
#else
                    tile_regs_wait();
#endif
                    for (uint32_t i = 0; i < SBW; ++i) pack_tile<true>(i, pass_cb, il0 + i);
                    tile_regs_release();
                }
                cb_push_back(pass_cb, 32);
            }
            const uint32_t ocb = GINSPLIT && !(q & 1) ? out_rd_cb : out_cb;
            cb_wait_front(p_cb, 32);
            cb_wait_front(g_cb, 32);
            cb_reserve_back(ocb, 32);
            reconfig_data_format(p_cb, g_cb);
            pack_reconfig_data_format(ocb);
            for (uint32_t t0 = 0; t0 < 32; t0 += 4) {
                tile_regs_acquire();
                if constexpr (ABL & 2) {
                } else if (MASK && q < CT2 / 2) {
                    mul_bcast_rows_init_short(g_cb, m_cb);
                    for (uint32_t i = 0; i < 4; ++i) mul_tiles_bcast_rows(g_cb, m_cb, t0 + i, t0 + i, i);
                    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        p_cb);
                    for (uint32_t i = 0; i < 4; ++i)
                        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                            p_cb, t0 + i, i);
                } else {
                    mul_tiles_init(p_cb, g_cb);
                    for (uint32_t i = 0; i < 4; ++i) mul_tiles(p_cb, g_cb, t0 + i, t0 + i, i);
                }
                if constexpr (RNE & 1) {
                    for (uint32_t i = 0; i < 4; ++i) round_bf16_rne_tile(i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < 4; ++i) pack_tile(i, ocb);
                tile_regs_release();
            }
            cb_push_back(ocb, 32);
            cb_pop_front(p_cb, 32);
            cb_pop_front(g_cb, 32);
        }
        cb_pop_front(x_cb, 32 * Kt);
        if constexpr (MASK) cb_pop_front(m_cb, 32);
    }
}
