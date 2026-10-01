// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_transpose writer. Output tile q of a unit, row r, is input tile r's row q: a tile row is
// two face rows (columns 0-15 in face 2*(row/16), 16-31 in the face after it), each ROW bytes, so
// the shuffle is 2048 local L1->L1 NoC reads a unit into a scratch c_16, then 32 whole-tile
// DRAM writes. Every element is moved, none is computed: bit-exact by construction.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t S1t = get_compile_time_arg_val(0);
    constexpr uint32_t S2t = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr uint32_t ROW = get_compile_time_arg_val(3);   // bytes in one face row: 16 elements
    constexpr auto d_args = TensorAccessorArgs<4>();
    const auto d = TensorAccessor(d_args, dst_addr);

    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t num = get_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t FACE = 16 * ROW;

    cb_reserve_back(cb_out, 32);
    const uint32_t scratch = get_write_ptr(cb_out);

    for (uint32_t u = first; u < first + num; ++u) {
        const uint32_t ct = u % Ct;
        const uint32_t It = (u / Ct) % S1t;
        const uint32_t Jt = u / (Ct * S1t);
        cb_wait_front(cb_in, 32);
        const uint32_t in = get_read_ptr(cb_in);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t src_tile = in + r * tile_bytes;
            const uint32_t dst_row = ((r >> 4) * 2) * FACE + (r & 15) * ROW;
            for (uint32_t q = 0; q < 32; ++q) {
                const uint32_t src = src_tile + ((q >> 4) * 2) * FACE + (q & 15) * ROW;
                const uint32_t dst = scratch + q * tile_bytes + dst_row;
                noc_async_read(get_noc_addr(src), dst, ROW);
                noc_async_read(get_noc_addr(src + FACE), dst + FACE, ROW);
            }
        }
        noc_async_read_barrier();
        cb_pop_front(cb_in, 32);
        uint32_t r = scratch;
        for (uint32_t q = 0; q < 32; ++q) {
            noc_async_write(r, d.get_noc_addr(((Jt * 32 + q) * S1t + It) * Ct + ct), tile_bytes);
            r += tile_bytes;
        }
        noc_async_write_barrier();
    }
}
