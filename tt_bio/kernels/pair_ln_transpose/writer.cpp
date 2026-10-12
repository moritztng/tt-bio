// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_ln_transpose writer: output tile (j = Jt*32 + q, It, ct) has row r = row q of normed tile (r, ct),
// the face-row shuffle of kernels/pair_transpose (local L1 -> L1 NoC reads into a 32-tile scratch, one
// channel tile at a time), then written to page (j * S1t + It) * Kt + ct of the [S2, S1, C] output.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t S1t = get_compile_time_arg_val(1);
    constexpr uint32_t S2t = get_compile_time_arg_val(2);
    constexpr uint32_t ROW = get_compile_time_arg_val(3);
    constexpr auto d_args = TensorAccessorArgs<4>();
    const auto d = TensorAccessor(d_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    if (nunits == 0) {
        return;
    }
    constexpr uint32_t cb_n = 16, cb_s = 24;
    constexpr uint32_t FACE = 16 * ROW;
    const uint32_t tb = get_tile_size(cb_n);
    cb_reserve_back(cb_s, 32);
    const uint32_t scratch = get_write_ptr(cb_s);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t It = u / S2t;
        const uint32_t Jt = u % S2t;
        cb_wait_front(cb_n, 32 * Kt);
        const uint32_t in = get_read_ptr(cb_n);
        for (uint32_t ct = 0; ct < Kt; ++ct) {
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t src_tile = in + (r * Kt + ct) * tb;
                const uint32_t dst_row = ((r >> 4) * 2) * FACE + (r & 15) * ROW;
                for (uint32_t q = 0; q < 32; ++q) {
                    const uint32_t src = src_tile + ((q >> 4) * 2) * FACE + (q & 15) * ROW;
                    const uint32_t dst = scratch + q * tb + dst_row;
                    noc_async_read(get_noc_addr(src), dst, ROW);
                    noc_async_read(get_noc_addr(src + FACE), dst + FACE, ROW);
                }
            }
            noc_async_read_barrier();
            for (uint32_t q = 0; q < 32; ++q) {
                noc_async_write(scratch + q * tb, d.get_noc_addr(((Jt * 32 + q) * S1t + It) * Kt + ct), tb);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_n, 32 * Kt);
    }
}
