// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_build reader. Unit u = (row bi = b * N + i, 32-pair block jt) owns output pages u * KT .. u * KT + KT - 1 of
// z [B, N, N, C]. Per unit it sends compute
//   c_0: the KT const tiles (page u * KT + ct of the constant projection, float32),
//   c_1: the KT gathered tiles, W[bins[bi, jt * 32 + r], ct * 32 + c] written straight into the tile's faces from the
//        float32 table held in L1, so every gathered value is the table's bit pattern,
//   c_2: one scale tile, scale[bi, jt * 32 + r] in every column of row r,
// and, whenever b changes, c_3 / c_4: the KT wg and beta tiles of design b (each row the same [C] vector).
// Common runtime args: const, table, bins, scale, wg, beta addresses. Per core: first unit, unit count.
#include "api/dataflow/dataflow_api.h"

namespace {
// Element (r, c) of a 32x32 tile: four 16x16 faces, row-major inside each, faces in (top-left, top-right,
// bottom-left, bottom-right) order.
inline uint32_t face_row(uint32_t r, uint32_t half) { return ((r >> 4) * 2 + half) * 256 + (r & 15) * 16; }
}  // namespace

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);    // padded tokens
    constexpr uint32_t NT = get_compile_time_arg_val(1);   // N / 32
    constexpr uint32_t KT = get_compile_time_arg_val(2);   // C / 32
    constexpr uint32_t NB = get_compile_time_arg_val(3);   // table rows (bins)
    constexpr uint32_t CU = get_compile_time_arg_val(4);   // units the constant holds: N * NT, or B * N * NT
    constexpr auto c_args = TensorAccessorArgs<5>();
    constexpr auto t_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t C = KT * 32;
    constexpr uint32_t ROW = N * 4;
    constexpr uint32_t cb_c = 0, cb_g = 1, cb_s = 2, cb_wg = 3, cb_beta = 4, cb_tab = 5, cb_bins = 6, cb_srow = 7;
    const auto sc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(0));
    const auto st = TensorAccessor(t_args, get_common_arg_val<uint32_t>(1), C * 4);
    const auto sb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(2), ROW);
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(3), ROW);
    const auto sw = TensorAccessor(w_args, get_common_arg_val<uint32_t>(4));
    const auto se = TensorAccessor(e_args, get_common_arg_val<uint32_t>(5));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    const uint32_t ts = get_tile_size(cb_c);

    const uint32_t tab_l1 = get_write_ptr(cb_tab);
    for (uint32_t k = 0; k < NB; ++k) {
        noc_async_read(st.get_noc_addr(k), tab_l1 + k * C * 4, C * 4);
    }
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab_l1);
    const uint32_t bins_l1 = get_write_ptr(cb_bins), srow_l1 = get_write_ptr(cb_srow);
    const volatile tt_l1_ptr uint32_t* bins = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bins_l1);
    const volatile tt_l1_ptr uint32_t* srow = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(srow_l1);

    uint32_t row = 0xFFFFFFFFu, design = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t bi = u / NT, jt = u % NT, b = bi / N;
        if (b != design) {
            cb_reserve_back(cb_wg, KT);
            cb_reserve_back(cb_beta, KT);
            uint32_t ww = get_write_ptr(cb_wg), we = get_write_ptr(cb_beta);
            for (uint32_t ct = 0; ct < KT; ++ct, ww += ts, we += ts) {
                noc_async_read_page(b * KT + ct, sw, ww);
                noc_async_read_page(b * KT + ct, se, we);
            }
            noc_async_read_barrier();
            cb_push_back(cb_wg, KT);
            cb_push_back(cb_beta, KT);
            design = b;
        }
        if (bi != row) {
            noc_async_read(sb.get_noc_addr(bi), bins_l1, ROW);
            noc_async_read(ss.get_noc_addr(bi), srow_l1, ROW);
            noc_async_read_barrier();
            row = bi;
        }
        cb_reserve_back(cb_c, KT);
        uint32_t wc = get_write_ptr(cb_c);
        for (uint32_t ct = 0; ct < KT; ++ct, wc += ts) {
            noc_async_read_page((u % CU) * KT + ct, sc, wc);
        }
        // the scale and gathered tiles are built while the const reads are in flight
        cb_reserve_back(cb_s, 1);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_s));
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t v = srow[jt * 32 + r];
            for (uint32_t half = 0; half < 2; ++half) {
                volatile tt_l1_ptr uint32_t* d = s + face_row(r, half);
                for (uint32_t c = 0; c < 16; ++c) {
                    d[c] = v;
                }
            }
        }
        cb_push_back(cb_s, 1);
        cb_reserve_back(cb_g, KT);
        volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_g));
        for (uint32_t r = 0; r < 32; ++r) {
            uint32_t k = bins[jt * 32 + r];
            k = k < NB ? k : NB - 1;
            const volatile tt_l1_ptr uint32_t* src = tab + k * C;
            for (uint32_t ct = 0; ct < KT; ++ct, src += 32) {
                volatile tt_l1_ptr uint32_t* d = g + ct * 1024;
                volatile tt_l1_ptr uint32_t* d0 = d + face_row(r, 0);
                volatile tt_l1_ptr uint32_t* d1 = d + face_row(r, 1);
                for (uint32_t c = 0; c < 16; ++c) {
                    d0[c] = src[c];
                    d1[c] = src[16 + c];
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_c, KT);
        cb_push_back(cb_g, KT);
    }
}
