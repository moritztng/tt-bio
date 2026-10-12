// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail, ending node: the update goes straight into z's orientation, so neither the
// update's pair_transpose nor its transpose_add runs. The attention ran on z^T: o and g are head-major
// [A, H, B, 32] over (a, b), and the update u[a, b] belongs to z[b, a]. A block is (bt, At): the 32 a of
// At by the 32 b of bt. The reader takes its 32 * Kt o tiles and 32 * Kt g tiles (a, h, bt) once, then
// emits 32 units, one per b: unit q's head tile h has row r = row q of raw tile (a = At*32 + r, h), the
// same face-row shuffle as kernels/pair_transpose. With RESID the unit's z tiles are (b, At, n). Compute
// is unchanged: it sees ordinary units whose rows are a instead of b.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);   // b tiles (o's third axis)
    constexpr uint32_t RESID = get_compile_time_arg_val(3);
    constexpr uint32_t At_n = get_compile_time_arg_val(4); // a tiles (o's first axis / 32)
    constexpr uint32_t ROW = get_compile_time_arg_val(5);  // bytes in one face row
    constexpr auto o_args = TensorAccessorArgs<6>();
    constexpr auto g_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1));
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2));
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(3));
    const uint32_t blk0 = get_arg_val<uint32_t>(0);
    const uint32_t nblk = get_arg_val<uint32_t>(1);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t o_cb = tt::CBIndex::c_1;
    constexpr uint32_t g_cb = tt::CBIndex::c_2;
    constexpr uint32_t z_cb = tt::CBIndex::c_3;
    constexpr uint32_t raw_cb = tt::CBIndex::c_8;          // reader scratch: 32 * Kt o tiles, then 32 * Kt g
    constexpr uint32_t FACE = 16 * ROW;
    const uint32_t tb = get_tile_size(o_cb);

    cb_reserve_back(w_cb, Kt * Nt);
    uint32_t wp = get_write_ptr(w_cb);
    for (uint32_t t = 0; t < Kt * Nt; ++t, wp += tb) noc_async_read_page(t, w, wp);
    noc_async_read_barrier();
    cb_push_back(w_cb, Kt * Nt);

    cb_reserve_back(raw_cb, 64 * Kt);
    const uint32_t raw_o = get_write_ptr(raw_cb);
    const uint32_t raw_g = raw_o + 32 * Kt * tb;

    for (uint32_t blk = blk0; blk < blk0 + nblk; ++blk) {
        const uint32_t bt = blk / At_n;
        const uint32_t At = blk % At_n;
        uint32_t op = raw_o, gp = raw_g;
        for (uint32_t r = 0; r < 32; ++r) {
            uint32_t page = ((At * 32 + r) * Kt) * St + bt;
            for (uint32_t h = 0; h < Kt; ++h, page += St, op += tb, gp += tb) {
                noc_async_read_page(page, o, op);
                noc_async_read_page(page, g, gp);
            }
        }
        noc_async_read_barrier();
        for (uint32_t q = 0; q < 32; ++q) {
            cb_reserve_back(g_cb, Kt);
            cb_reserve_back(o_cb, Kt);
            const uint32_t gd = get_write_ptr(g_cb), od = get_write_ptr(o_cb);
            const uint32_t src_row = ((q >> 4) * 2) * FACE + (q & 15) * ROW;
            for (uint32_t h = 0; h < Kt; ++h) {
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t src = (r * Kt + h) * tb + src_row;
                    const uint32_t dst = h * tb + ((r >> 4) * 2) * FACE + (r & 15) * ROW;
                    noc_async_read(get_noc_addr(raw_o + src), od + dst, ROW);
                    noc_async_read(get_noc_addr(raw_o + src + FACE), od + dst + FACE, ROW);
                    noc_async_read(get_noc_addr(raw_g + src), gd + dst, ROW);
                    noc_async_read(get_noc_addr(raw_g + src + FACE), gd + dst + FACE, ROW);
                }
            }
            if constexpr (RESID) {
                cb_reserve_back(z_cb, Nt);
                uint32_t zp = get_write_ptr(z_cb);
                const uint32_t u = (bt * 32 + q) * At_n + At;
                for (uint32_t n = 0; n < Nt; ++n, zp += tb) noc_async_read_page(u * Nt + n, z, zp);
            }
            noc_async_read_barrier();
            cb_push_back(g_cb, Kt);
            cb_push_back(o_cb, Kt);
            if constexpr (RESID) cb_push_back(z_cb, Nt);
        }
    }
}
