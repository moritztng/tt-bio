// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail, ending variant: z[i, j] += u[j, i] in place, u never written out. A group is
// 32 rows b = j of the transposed update at one column tile jt (32 i): its 32 x Nt tiles stay in c_16.
// Per channel tile n the writer reads the 32 z tiles (i = jt*32 + q, column tile b / 32, n) the group
// lands on, shuffles the group's tile n into z's orientation as pair_transpose_add's writer does
// (row q of unit r's tile -> row r of tile q, two 32-byte face rows each), hands both to compute and
// writes the sums back to the pages they were read from. Each z page belongs to one group.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);     // column tiles of u = row tiles of z
    constexpr uint32_t Bt = get_compile_time_arg_val(2);     // row tiles of u = column tiles of z
    constexpr auto z_args = TensorAccessorArgs<3>();
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(0));
    const uint32_t g0 = get_arg_val<uint32_t>(0);
    const uint32_t ngroups = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_16, z_cb = tt::CBIndex::c_3;
    constexpr uint32_t sum_cb = tt::CBIndex::c_17, slot_cb = tt::CBIndex::c_18;
    constexpr uint32_t ROW = 32, FACE = 16 * ROW;
    const uint32_t tb = get_tile_size(out_cb);
    auto row = [](uint32_t r) { return ((r >> 4) * 2) * FACE + (r & 15) * ROW; };

    for (uint32_t g = g0; g < g0 + ngroups; ++g) {
        const uint32_t bt = g / St, jt = g % St;
        auto page = [&](uint32_t q, uint32_t n) { return ((jt * 32 + q) * Bt + bt) * Nt + n; };
        cb_wait_front(out_cb, 32 * Nt);
        const uint32_t in = get_read_ptr(out_cb);
        for (uint32_t n = 0; n < Nt; ++n) {
            cb_reserve_back(z_cb, 32);
            uint32_t lz = get_write_ptr(z_cb);
            for (uint32_t q = 0; q < 32; ++q, lz += tb) noc_async_read_page(page(q, n), z, lz);
            cb_reserve_back(slot_cb, 32);
            const uint32_t slot = get_write_ptr(slot_cb);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t src_tile = in + (r * Nt + n) * tb;
                const uint32_t dst_row = row(r);
                for (uint32_t q = 0; q < 32; ++q) {
                    const uint32_t src = src_tile + row(q);
                    const uint32_t dst = slot + q * tb + dst_row;
                    noc_async_read(get_noc_addr(src), dst, ROW);
                    noc_async_read(get_noc_addr(src + FACE), dst + FACE, ROW);
                }
            }
            noc_async_read_barrier();
            cb_push_back(z_cb, 32);
            cb_push_back(slot_cb, 32);
            for (uint32_t q = 0; q < 32; q += 4) {
                cb_wait_front(sum_cb, 4);
                uint32_t l1 = get_read_ptr(sum_cb);
                for (uint32_t k = 0; k < 4; ++k, l1 += tb) noc_async_write_page(page(q + k, n), z, l1);
                noc_async_writes_flushed();
                cb_pop_front(sum_cb, 4);
            }
        }
        cb_pop_front(out_cb, 32 * Nt);
    }
    noc_async_write_barrier();
}
