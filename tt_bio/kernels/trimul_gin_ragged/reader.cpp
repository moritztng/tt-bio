// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul_gin_moved's reader for a ragged pair grid (S not a whole number of tiles, AF2 at 195) with
// the projection biases. The x tensor [B?, S, S, K] tiles only its last two axes, so a row x is
// St * Kt pages and the x axis is not padded: row tiles past S re-read row S - 1 (the writer zeroes
// their output rows). A unit is (cube, part): part splits the channel tiles CT2 into QS ranges so a
// small grid still fills the cores; each unit reloads its 32 * Kt activation tiles. The bias tiles
// [bp tiles | bg tiles] (2 * CT2, each row c holding bias[c] in every column) go to c_7 once.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t CT2 = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t MASK = get_compile_time_arg_val(4);
    constexpr uint32_t QS = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto wp_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wg_args = TensorAccessorArgs<wp_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto wp = TensorAccessor(wp_args, get_common_arg_val<uint32_t>(1));
    const auto wg = TensorAccessor(wg_args, get_common_arg_val<uint32_t>(2));
    const auto mk = TensorAccessor(m_args, get_common_arg_val<uint32_t>(3));
    const auto bs = TensorAccessor(b_args, get_common_arg_val<uint32_t>(4));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t x_cb = tt::CBIndex::c_1;
    constexpr uint32_t mscratch_cb = tt::CBIndex::c_3;
    constexpr uint32_t m_cb = tt::CBIndex::c_6;
    constexpr uint32_t b_cb = tt::CBIndex::c_7;
    const uint32_t tb = get_tile_size(x_cb);

    cb_reserve_back(w_cb, 2 * CT2 * Kt);
    uint32_t w = get_write_ptr(w_cb);
    for (uint32_t t = 0; t < CT2 * Kt; ++t, w += tb) noc_async_read_page(t, wp, w);
    for (uint32_t t = 0; t < CT2 * Kt; ++t, w += tb) noc_async_read_page(t, wg, w);
    cb_reserve_back(b_cb, 2 * CT2);
    w = get_write_ptr(b_cb);
    for (uint32_t t = 0; t < 2 * CT2; ++t, w += tb) noc_async_read_page(t, bs, w);
    noc_async_read_barrier();
    cb_push_back(w_cb, 2 * CT2 * Kt);
    cb_push_back(b_cb, 2 * CT2);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t cu = u / QS;
        const uint32_t b = cu / (St * St);
        const uint32_t xt = (cu / St) % St;
        const uint32_t yt = cu % St;
        cb_reserve_back(x_cb, 32 * Kt);
        const uint32_t base = get_write_ptr(x_cb);
        for (uint32_t il = 0; il < 32; ++il) {
            const uint32_t xr = xt * 32 + il < S ? xt * 32 + il : S - 1;
            const uint32_t page = ((b * S + xr) * St + yt) * Kt;
            uint32_t dst = base + il * tb;
            for (uint32_t dt = 0; dt < Kt; ++dt, dst += 32 * tb) noc_async_read_page(page + dt, x, dst);
        }
        noc_async_read_barrier();
        cb_push_back(x_cb, 32 * Kt);
        if constexpr (MASK) {
            const uint32_t scratch = get_write_ptr(mscratch_cb);
            noc_async_read_page((b * St + xt) * St + yt, mk, scratch);
            noc_async_read_barrier();
            cb_reserve_back(m_cb, 32);
            uint32_t d = get_write_ptr(m_cb);
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            // Row il of a bf16 tile into row 0 of tile il, as in trimul_gin_moved's reader.
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
