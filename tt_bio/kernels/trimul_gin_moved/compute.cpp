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

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
// sigmoid(x) = 0.5 + sign(x) * h(t), t = min(|x| / 9, 1), h a degree-8 polynomial with h(0) = 0 fitted
// (weighted minimax) to sigmoid(9 t) - 0.5 on [0, 1]. Max |error| 3.0e-4 over all x in float32 (bf16's
// half-ULP is 1-2e-3 above 0.25), output inside (0, 1). ~25 SFPU instructions against ~55 for
// exp_21f + reciprocal, and the sigmoid was 3.3 of gin_moved's 7.3 ms at 736 on WH (stage ablation).
template <int ITERATIONS = 8>
inline void _sigmoid_poly_() {
    const sfpi::vFloat c8 = 9.57322883605957f, c7 = -51.5362548828125f, c6 = 114.75630187988281f,
                       c5 = -135.66062927246094f, c4 = 89.25188446044922f, c3 = -29.09324073791504f,
                       c2 = 0.9815341234207153f, c1 = 2.2268805503845215f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat t = sfpi::abs(x) * 0.1111111119389534f;
        v_if(t > 1.0f) { t = 1.0f; }
        v_endif;
        sfpi::vFloat h = c8 * t + c7;
        h = h * t + c6;
        h = h * t + c5;
        h = h * t + c4;
        h = h * t + c3;
        h = h * t + c2;
        h = h * t + c1;
        h = h * t;
        sfpi::dst_reg[0] = sfpi::setsgn(h, x) + 0.5f;
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_MATH

ALWI void sigmoid_poly_tile(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_sigmoid_poly_<8>, idst, (int)VectorMode::RC)));
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t CT2 = get_compile_time_arg_val(1);
    constexpr uint32_t SBW = get_compile_time_arg_val(2);
    constexpr uint32_t MASK = get_compile_time_arg_val(3);
    // Stage ablation, diagnostic only: bit 0 no sigmoid, bit 1 no gate multiply, bit 4 no matmul.
    constexpr uint32_t ABL = get_compile_time_arg_val(4);
    constexpr uint32_t SIGPOLY = get_compile_time_arg_val(5);   // 1: the polynomial sigmoid above
    const uint32_t nunits = get_arg_val<uint32_t>(0);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t x_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t p_cb = tt::CBIndex::c_4;
    constexpr uint32_t g_cb = tt::CBIndex::c_5;
    constexpr uint32_t m_cb = tt::CBIndex::c_6;

    sigmoid_tile_init();
    mm_init(w_cb, x_cb, p_cb, 1);
    cb_wait_front(w_cb, 2 * CT2 * Kt);

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
                    if (pass == 1 && !(ABL & 1)) {
                        for (uint32_t i = 0; i < SBW; ++i) {
                            if constexpr (SIGPOLY) sigmoid_poly_tile(i);
                            else sigmoid_bf16_tile(i);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < SBW; ++i) pack_tile<true>(i, pass_cb, il0 + i);
                    tile_regs_release();
                }
                cb_push_back(pass_cb, 32);
            }
            cb_wait_front(p_cb, 32);
            cb_wait_front(g_cb, 32);
            cb_reserve_back(out_cb, 32);
            reconfig_data_format(p_cb, g_cb);
            pack_reconfig_data_format(out_cb);
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
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < 4; ++i) pack_tile(i, out_cb);
                tile_regs_release();
            }
            cb_push_back(out_cb, 32);
            cb_pop_front(p_cb, 32);
            cb_pop_front(g_cb, 32);
        }
        cb_pop_front(x_cb, 32 * Kt);
        if constexpr (MASK) cb_pop_front(m_cb, 32);
    }
}
