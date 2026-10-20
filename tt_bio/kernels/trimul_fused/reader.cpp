// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul_fused reader (bc4-bet-fusedbuild). Phase 1 of the fused AF2 trimul forward: a unit is a half cube, rows
// r = 32 rt + 16 h + li (li < 16) of the pair grid at column tile st. Once: the four transposed in-projections W^T
// (tile (p * 4 + q) * Kt + kc, p = a_p, a_g, b_p, b_g) into c_0, their bias tiles (row c = bias[c]) into c_3,
// LayerNorm gamma / beta (row 0) into c_8 / c_9, the 1/K scaler into c_10 and an fp32 eps tile into c_11. Per unit:
// the 16 row tiles of z, Kt pages each, into c_1, and (MASK) 16 mask tiles into c_4, tile li carrying mask row
// r in all 32 rows so compute can multiply it in DST.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t MASK = get_compile_time_arg_val(2);
    constexpr uint32_t SCALER = get_compile_time_arg_val(3);
    constexpr uint32_t EPS = get_compile_time_arg_val(4);
    // ABL (bench only): bit 5 (32) skips the weight reads, bit 6 (64) the z reads, bit 7 (128) the mask build;
    // the CBs are pushed as they are.
    constexpr uint32_t ABL = get_compile_time_arg_val(5);
    constexpr auto z_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<bt_args.next_compile_time_args_offset()>();
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(0));
    const auto wt = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1));
    const auto bs = TensorAccessor(b_args, get_common_arg_val<uint32_t>(2));
    const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(3));
    const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(4));
    const auto mk = TensorAccessor(m_args, get_common_arg_val<uint32_t>(5));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_w = 0, cb_z = 1, cb_bias = 3, cb_mask = 4, cb_gamma = 8, cb_beta = 9, cb_scaler = 10,
                       cb_eps = 11, cb_mscr = 7;
    constexpr uint32_t NW = 16 * Kt;   // 4 projections x 4 channel tiles x Kt
    const uint32_t tb = get_tile_size(cb_z);

    generate_reduce_scaler(cb_scaler, SCALER);
    {
        cb_reserve_back(cb_eps, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(cb_eps, 1);
    }
    cb_reserve_back(cb_w, NW);
    cb_reserve_back(cb_bias, 16);
    cb_reserve_back(cb_gamma, Kt);
    cb_reserve_back(cb_beta, Kt);
    uint32_t w = get_write_ptr(cb_w);
    for (uint32_t t = 0; t < (ABL & 32 ? 0 : NW); ++t, w += tb) noc_async_read_page(t, wt, w);
    w = get_write_ptr(cb_bias);
    for (uint32_t t = 0; t < 16; ++t, w += tb) noc_async_read_page(t, bs, w);
    w = get_write_ptr(cb_gamma);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, gm, w);
    w = get_write_ptr(cb_beta);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, bt, w);
    noc_async_read_barrier();
    cb_push_back(cb_w, NW);
    cb_push_back(cb_bias, 16);
    cb_push_back(cb_gamma, Kt);
    cb_push_back(cb_beta, Kt);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t h = u % 2, cube = u / 2, rt = cube / St, st = cube % St;
        if constexpr (MASK && (ABL & 128)) {
            cb_reserve_back(cb_mask, 16);
            cb_push_back(cb_mask, 16);
        } else if constexpr (MASK) {
            const uint32_t scr = get_write_ptr(cb_mscr);
            noc_async_read_page(rt * St + st, mk, scr);
            noc_async_read_barrier();
            cb_reserve_back(cb_mask, 16);
            uint32_t d = get_write_ptr(cb_mask);
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
            for (uint32_t li = 0; li < 16; ++li, d += tb) {
                // source row 16 h + li sits in faces 2h (cols 0-15) and 2h + 1 (cols 16-31), row li of each.
                const uint32_t s0 = (2 * h * 512 + li * 32) / 4;
                volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d);
                for (uint32_t f = 0; f < 4; f += 2) {
                    for (uint32_t rr = 0; rr < 16; ++rr) {
                        const uint32_t o = (f * 512 + rr * 32) / 4;
                        for (uint32_t k = 0; k < 8; ++k) {
                            dst[o + k] = src[s0 + k];
                            dst[o + 128 + k] = src[s0 + 128 + k];
                        }
                    }
                }
            }
            cb_push_back(cb_mask, 16);
        }
        for (uint32_t li = 0; li < 16; ++li) {
            const uint32_t r = rt * 32 + h * 16 + li;
            cb_reserve_back(cb_z, Kt);
            uint32_t dst = get_write_ptr(cb_z);
            for (uint32_t kc = 0; kc < (ABL & 64 ? 0 : Kt); ++kc, dst += tb) noc_async_read_page((r * St + st) * Kt + kc, z, dst);
            noc_async_read_barrier();
            cb_push_back(cb_z, Kt);
        }
    }
}
