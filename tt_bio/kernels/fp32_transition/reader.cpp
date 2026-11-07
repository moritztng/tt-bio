// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fp32_transition reader (tt_bio/fp32_transition.py): gamma and beta (KT fp32 tiles each) and token 0's x row copied
// over all 32 rows of KT tiles (c_8) once; per row tile r of this core's nr from r0 its KT x tiles (c_0), the two LN
// statistics broadcast back over the columns (c_4 -> c_5, c_6 -> c_7, as fp32_ln's reader; twice for token 0 first),
// then column n of w1 and of w2 (KT tiles each) for each of this core's NHG hidden tiles from j * NHG, and column m
// of w3 over the same K run in KC3-tile chunks.
#include "api/dataflow/dataflow_api.h"

// dst[row][col] = src[row][0] for a 32 x 32 fp32 tile in face order (four 16 x 16 faces, row-major in each).
void broadcast_col0(uint32_t src_cb, uint32_t dst_cb) {
    cb_wait_front(src_cb, 1);
    cb_reserve_back(dst_cb, 1);
    auto* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(src_cb));
    auto* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(dst_cb));
    for (uint32_t i = 0; i < 32; ++i) {
        const uint32_t left = (i < 16 ? 0 : 512) + (i & 15) * 16;
        const uint32_t v = s[left];
        for (uint32_t j = 0; j < 16; ++j) {
            d[left + j] = v;
            d[left + 256 + j] = v;
        }
    }
    cb_push_back(dst_cb, 1);
    cb_pop_front(src_cb, 1);
}

// rows 1..31 of an fp32 tile in face order <- its row 0 (face 0 and face 1 hold row 0's two 16-column halves).
void broadcast_row0(uint32_t addr) {
    auto* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 1; i < 32; ++i) {
        const uint32_t row = (i < 16 ? 0 : 512) + (i & 15) * 16;
        for (uint32_t j = 0; j < 16; ++j) {
            t[row + j] = t[j];
            t[row + 256 + j] = t[256 + j];
        }
    }
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NHG = get_compile_time_arg_val(2);
    constexpr uint32_t KC3 = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto w1_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto w3_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto gm = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1));
    const auto bt = TensorAccessor(b_args, get_common_arg_val<uint32_t>(2));
    const auto w1 = TensorAccessor(w1_args, get_common_arg_val<uint32_t>(3));
    const auto w2 = TensorAccessor(w2_args, get_common_arg_val<uint32_t>(4));
    const auto w3 = TensorAccessor(w3_args, get_common_arg_val<uint32_t>(5));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nr = get_arg_val<uint32_t>(1);
    const uint32_t h0 = get_arg_val<uint32_t>(2) * NHG;

    constexpr uint32_t cb_x = 0, cb_g = 1, cb_b = 2, cb_x0 = 8, cb_w1 = 11, cb_w2 = 12, cb_w3 = 13;
    const uint32_t tb = get_tile_size(cb_x), wb = get_tile_size(cb_w1), w3b = get_tile_size(cb_w3);

    cb_reserve_back(cb_g, KT);
    cb_reserve_back(cb_b, KT);
    cb_reserve_back(cb_x0, KT);
    uint32_t ag = get_write_ptr(cb_g), ab = get_write_ptr(cb_b), a0 = get_write_ptr(cb_x0);
    for (uint32_t t = 0; t < KT; ++t) {
        noc_async_read_page(t, gm, ag + t * tb);
        noc_async_read_page(t, bt, ab + t * tb);
        noc_async_read_page(t, x, a0 + t * tb);
    }
    noc_async_read_barrier();
    for (uint32_t t = 0; t < KT; ++t) broadcast_row0(a0 + t * tb);
    cb_push_back(cb_g, KT);
    cb_push_back(cb_b, KT);
    cb_push_back(cb_x0, KT);
    broadcast_col0(tt::CBIndex::c_4, tt::CBIndex::c_5);
    broadcast_col0(tt::CBIndex::c_6, tt::CBIndex::c_7);

    for (uint32_t r = r0; r < r0 + nr; ++r) {
        cb_reserve_back(cb_x, KT);
        uint32_t w = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < KT; ++t, w += tb) noc_async_read_page(r * KT + t, x, w);
        noc_async_read_barrier();
        cb_push_back(cb_x, KT);
        broadcast_col0(tt::CBIndex::c_4, tt::CBIndex::c_5);
        broadcast_col0(tt::CBIndex::c_6, tt::CBIndex::c_7);
        for (uint32_t n = h0; n < h0 + NHG; ++n) {
            cb_reserve_back(cb_w1, KT);
            cb_reserve_back(cb_w2, KT);
            uint32_t p1 = get_write_ptr(cb_w1), p2 = get_write_ptr(cb_w2);
            for (uint32_t k = 0; k < KT; ++k, p1 += wb, p2 += wb) {
                noc_async_read_page(k * NH + n, w1, p1);
                noc_async_read_page(k * NH + n, w2, p2);
            }
            noc_async_read_barrier();
            cb_push_back(cb_w1, KT);
            cb_push_back(cb_w2, KT);
        }
        for (uint32_t m = 0; m < KT; ++m) {
            for (uint32_t k0 = h0; k0 < h0 + NHG; k0 += KC3) {
                cb_reserve_back(cb_w3, KC3);
                uint32_t p3 = get_write_ptr(cb_w3);
                for (uint32_t k = k0; k < k0 + KC3; ++k, p3 += w3b) noc_async_read_page(k * KT + m, w3, p3);
                noc_async_read_barrier();
                cb_push_back(cb_w3, KC3);
            }
        }
    }
}
