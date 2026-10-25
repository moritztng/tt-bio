// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail of the ending block, into z^T: z[s, b] += u[b, s] without writing u. A unit is
// a 32 x 32 block (bt, st): compute leaves its 32 row tiles (bt*32 + r, st), Nt channel tiles each, in
// c_16 ([r][n], all 32*Nt pages). Per channel tile n this kernel reads the 32 z tiles the block lands
// on (z page (s*Bt + bt)*Nt + n, s = st*32 + q) into c_3, shuffles row q of staged tile (r, n) into row r
// of tile q of c_18 (pair_transpose_add's shuffle, two 32 B face rows per move), and writes the 32 sums
// compute returns in c_17 back to the z pages they were read from. Each z page belongs to one unit.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t GRAN = get_compile_time_arg_val(3);
    constexpr auto z_args = TensorAccessorArgs<4>();
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0) >> 5;
    const uint32_t nunits = get_arg_val<uint32_t>(1) >> 5;

    constexpr uint32_t stage_cb = tt::CBIndex::c_16, z_cb = tt::CBIndex::c_3;
    constexpr uint32_t t_cb = tt::CBIndex::c_18, s_cb = tt::CBIndex::c_17;
    constexpr uint32_t ROW = 16 * 2, FACE = 16 * ROW;
    const uint32_t tb = get_tile_size(stage_cb);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t bt = u / St, st = u % St;
        auto page = [&](uint32_t q, uint32_t n) { return ((st * 32 + q) * Bt + bt) * Nt + n; };
        cb_wait_front(stage_cb, 32 * Nt);
        const uint32_t stage = get_read_ptr(stage_cb);
        for (uint32_t n = 0; n < Nt; ++n) {
            cb_reserve_back(z_cb, 32);
            uint32_t lz = get_write_ptr(z_cb);
            for (uint32_t q = 0; q < 32; ++q, lz += tb) noc_async_read(z.get_noc_addr(page(q, n)), lz, tb);
            cb_reserve_back(t_cb, 32);
            const uint32_t slot = get_write_ptr(t_cb);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t src_tile = stage + (r * Nt + n) * tb;
                const uint32_t dst_row = ((r >> 4) * 2) * FACE + (r & 15) * ROW;
                for (uint32_t q = 0; q < 32; ++q) {
                    const uint32_t src = src_tile + ((q >> 4) * 2) * FACE + (q & 15) * ROW;
                    const uint32_t dst = slot + q * tb + dst_row;
                    noc_async_read(get_noc_addr(src), dst, ROW);
                    noc_async_read(get_noc_addr(src + FACE), dst + FACE, ROW);
                }
            }
            noc_async_read_barrier();
            cb_push_back(z_cb, 32);
            cb_push_back(t_cb, 32);
            for (uint32_t q = 0; q < 32; q += GRAN) {
                cb_wait_front(s_cb, GRAN);
                uint32_t l1 = get_read_ptr(s_cb);
                for (uint32_t k = 0; k < GRAN; ++k, l1 += tb) noc_async_write(l1, z.get_noc_addr(page(q + k, n)), tb);
                noc_async_writes_flushed();
                cb_pop_front(s_cb, GRAN);
            }
        }
        cb_pop_front(stage_cb, 32 * Nt);
    }
    noc_async_write_barrier();
}
