// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul input side T1 reader. Once: [Wp^T | Wg^T] into c_0 (as gin_moved), Wg_out [K, N] into c_18,
// gamma / beta (Kt tiles each, row 0 carries the [K] vector) into c_8 / c_9, the 1/K reduce scaler
// into c_10 and a full fp32 eps tile into c_11 (as T2). Then per unit (b, xt, yt) the 32 raw z row
// tiles R_il = (b, x = xt*32 + il, yt), il = 0..31, in il order, Kt tiles each, RB rows per barrier,
// into c_7: compute normalises them row by row. With MASK the unit's 32 row-broadcast mask tiles
// follow in c_6, cut exactly as gin_moved's reader cuts them. NEVER RUN ON A DEVICE YET.
// ABL (bench-only define, trimul_tail.T1_ABL; 0 in every model path): bits that skip one stage's
// work while keeping its circular-buffer traffic, to time each stage by difference (perf/spd_trimul/in_t1.py --abl).
//   1 layer_norm, 2 g_out matmul, 4 [a | b] matmuls, 8 gate multiply, 16 z reads, 32 writer gather, 64 writer writes.
#ifndef ABL
#define ABL 0
#endif
// T1_S (defined only for a ragged pair size S, S % 32 != 0, e.g. AF2-IG at 195): z is [B?, S, S, K]
// with only y and K tiled, so the x axis has S rows and the y axis St = ceil(S / 32) tiles. As
// kernels/trimul_gin_ragged's reader: row tiles x >= S re-read row S - 1 (the writer zeroes their a /
// b rows and skips their g_out rows). Additionally the padded rows y >= S of the last y tile are
// zeroed in L1 before compute sees them, so the LayerNorm of a padding row is beta (finite) whatever
// the padding of z holds; compute zeroes those columns of a / b after the gate with the column mask.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t CT2 = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t MASK = get_compile_time_arg_val(3);
    constexpr uint32_t NTG = get_compile_time_arg_val(4);      // g_out channel tiles, 0: none
    constexpr uint32_t RB = get_compile_time_arg_val(5);       // row tiles per read barrier
    constexpr uint32_t SCALER = get_compile_time_arg_val(6);   // packed bf16 pair, 1/K
    constexpr uint32_t EPS = get_compile_time_arg_val(7);      // fp32 bits
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto wp_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wg_args = TensorAccessorArgs<wp_args.next_compile_time_args_offset()>();
    constexpr auto wo_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<wo_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<bt_args.next_compile_time_args_offset()>();
    static_assert(32 % RB == 0, "RB must divide the 32 row tiles of a unit");

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto wp = TensorAccessor(wp_args, get_common_arg_val<uint32_t>(1));
    const auto wg = TensorAccessor(wg_args, get_common_arg_val<uint32_t>(2));
    const auto wo = TensorAccessor(wo_args, get_common_arg_val<uint32_t>(3));
    const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(4));
    const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(5));
    const auto mk = TensorAccessor(m_args, get_common_arg_val<uint32_t>(6));
#ifdef T1_BIAS
    // The projection biases (trimul_tail.t1_bias_tiles): [bp | bg] as gin_ragged_bias lays them
    // (2 * CT2 tiles, rows c carry bias[32 q + c]), its column-mask tile (used by compute only under
    // T1_S, a ragged S), then NTG g_out bias tiles (every row bias[32 n + col]). Read once into c_20.
    constexpr auto bias_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto bias = TensorAccessor(bias_args, get_common_arg_val<uint32_t>(7));
    constexpr uint32_t bias_cb = tt::CBIndex::c_20;
    constexpr uint32_t NBIAS = 2 * CT2 + 1 + NTG;
#endif
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t mscratch_cb = tt::CBIndex::c_3;
    constexpr uint32_t m_cb = tt::CBIndex::c_6;
    constexpr uint32_t x_cb = tt::CBIndex::c_7;
    constexpr uint32_t gamma_cb = tt::CBIndex::c_8;
    constexpr uint32_t beta_cb = tt::CBIndex::c_9;
    constexpr uint32_t scaler_cb = tt::CBIndex::c_10;
    constexpr uint32_t eps_cb = tt::CBIndex::c_11;
    constexpr uint32_t wo_cb = tt::CBIndex::c_18;
#ifdef T1_S
    constexpr uint32_t S = T1_S;
    static_assert(S > (St - 1) * 32 && S < St * 32, "T1_S is a ragged S of St tiles");
    constexpr uint32_t LAST = S - (St - 1) * 32;   // valid rows of the last y tile
#else
    constexpr uint32_t S = St * 32;
#endif
    const uint32_t tb = get_tile_size(x_cb);

    generate_reduce_scaler(scaler_cb, SCALER);
    {
        cb_reserve_back(eps_cb, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(eps_cb));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(eps_cb, 1);
    }
    cb_reserve_back(w_cb, 2 * CT2 * Kt);
    cb_reserve_back(gamma_cb, Kt);
    cb_reserve_back(beta_cb, Kt);
    uint32_t w = get_write_ptr(w_cb);
    for (uint32_t t = 0; t < CT2 * Kt; ++t, w += tb) noc_async_read_page(t, wp, w);
    for (uint32_t t = 0; t < CT2 * Kt; ++t, w += tb) noc_async_read_page(t, wg, w);
    w = get_write_ptr(gamma_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, gm, w);
    w = get_write_ptr(beta_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, bt, w);
    if constexpr (NTG) {
        cb_reserve_back(wo_cb, Kt * NTG);
        w = get_write_ptr(wo_cb);
        for (uint32_t t = 0; t < Kt * NTG; ++t, w += tb) noc_async_read_page(t, wo, w);
    }
#ifdef T1_BIAS
    cb_reserve_back(bias_cb, NBIAS);
    w = get_write_ptr(bias_cb);
    for (uint32_t t = 0; t < NBIAS; ++t, w += tb) noc_async_read_page(t, bias, w);
#endif
    noc_async_read_barrier();
    cb_push_back(w_cb, 2 * CT2 * Kt);
#ifdef T1_BIAS
    cb_push_back(bias_cb, NBIAS);
#endif
    cb_push_back(gamma_cb, Kt);
    cb_push_back(beta_cb, Kt);
    if constexpr (NTG) cb_push_back(wo_cb, Kt * NTG);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / (St * St);
        const uint32_t xt = (u / St) % St;
        const uint32_t yt = u % St;
        // Row tile (b, x, yt) of z is page ((b * S + x) * St + yt) * Kt + dt (trimul_tail.t1_x_page).
#ifdef T1_S
        for (uint32_t il0 = 0; il0 < 32; il0 += RB) {
            cb_reserve_back(x_cb, RB * Kt);
            const uint32_t base = get_write_ptr(x_cb);
            uint32_t dst = base;
            for (uint32_t r = 0; r < RB; ++r) {
                const uint32_t xr = xt * 32 + il0 + r < S ? xt * 32 + il0 + r : S - 1;
                const uint32_t page = ((b * S + xr) * St + yt) * Kt;
                for (uint32_t dt = 0; dt < Kt; ++dt, dst += tb)
                    if constexpr (!(ABL & 16)) noc_async_read_page(page + dt, x, dst);
            }
            noc_async_read_barrier();
            if (yt == St - 1) {
                // Rows y >= S of each bf16 tile: row r is 16 elements (8 words) in face (r / 16) * 2
                // and 8 more in the face after it, at word (r % 16) * 8 of each face.
                for (uint32_t t = 0; t < RB * Kt; ++t) {
                    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * tb);
                    for (uint32_t r = LAST; r < 32; ++r) {
                        const uint32_t w0 = (r >> 4) * 2 * 128 + (r & 15) * 8;
                        for (uint32_t k = 0; k < 8; ++k) {
                            p[w0 + k] = 0;
                            p[w0 + 128 + k] = 0;
                        }
                    }
                }
            }
            cb_push_back(x_cb, RB * Kt);
        }
#else
        uint32_t page = ((b * S + xt * 32) * St + yt) * Kt;
        for (uint32_t il0 = 0; il0 < 32; il0 += RB) {
            cb_reserve_back(x_cb, RB * Kt);
            uint32_t dst = get_write_ptr(x_cb);
            for (uint32_t r = 0; r < RB; ++r, page += St * Kt)
                for (uint32_t dt = 0; dt < Kt; ++dt, dst += tb)
                    if constexpr (!(ABL & 16)) noc_async_read_page(page + dt, x, dst);
            noc_async_read_barrier();
            cb_push_back(x_cb, RB * Kt);
        }
#endif
        if constexpr (MASK) {
            const uint32_t scratch = get_write_ptr(mscratch_cb);
            noc_async_read_page((b * St + xt) * St + yt, mk, scratch);
            noc_async_read_barrier();
            cb_reserve_back(m_cb, 32);
            uint32_t d = get_write_ptr(m_cb);
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            // Row il of a bf16 tile: 16 elements in face (il / 16) * 2 and 16 in the face after it,
            // row il % 16 of each; row 0 of the destination is the first row of faces 0 and 1.
            for (uint32_t il = 0; il < 32; ++il, d += tb) {
                volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d);
                const uint32_t s0 = (((il >> 4) * 2) * 256 + (il & 15) * 16) / 2;
                for (uint32_t k = 0; k < 8; ++k) {
                    dst[k] = src[s0 + k];
                    dst[128 + k] = src[s0 + 128 + k];
                }
            }
            cb_push_back(m_cb, 32);
        }
    }
}
