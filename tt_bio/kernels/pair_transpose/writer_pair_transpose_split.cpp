// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_transpose writer, split variant: shuffles output tiles QR..31 of each unit into its slot,
// starts their DRAM writes, then waits for the reader's tiles 0..QR-1 and writes those. Every
// element is moved, none is computed: bit-exact by construction, like the plain variant.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, cb_half = 1, cb_free = 2, cb_out = 16;
    constexpr uint32_t S1t = get_compile_time_arg_val(0);
    constexpr uint32_t S2t = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr uint32_t ROW = get_compile_time_arg_val(3);
    constexpr uint32_t QR = get_compile_time_arg_val(4);
    constexpr auto d_args = TensorAccessorArgs<5>();
    const auto d = TensorAccessor(d_args, dst_addr);

    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t num = get_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t FACE = 16 * ROW;
    const uint32_t out_base = get_write_ptr(cb_out);

    for (uint32_t i = 0; i < num; ++i) {
        const uint32_t u = first + i;
        const uint32_t ct = u % Ct;
        const uint32_t It = (u / Ct) % S1t;
        const uint32_t Jt = u / (Ct * S1t);
        const uint32_t slot = out_base + (i & 1) * 32 * tile_bytes;
        cb_wait_front(cb_in, 32);
        const uint32_t in = get_read_ptr(cb_in);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t src_tile = in + r * tile_bytes;
            const uint32_t dst_row = ((r >> 4) * 2) * FACE + (r & 15) * ROW;
            for (uint32_t q = QR; q < 32; ++q) {
                const uint32_t src = src_tile + ((q >> 4) * 2) * FACE + (q & 15) * ROW;
                const uint32_t dst = slot + q * tile_bytes + dst_row;
                noc_async_read(get_noc_addr(src), dst, ROW);
                noc_async_read(get_noc_addr(src + FACE), dst + FACE, ROW);
            }
        }
        noc_async_read_barrier();
        for (uint32_t q = QR; q < 32; ++q) {
            noc_async_write(slot + q * tile_bytes, d.get_noc_addr(((Jt * 32 + q) * S1t + It) * Ct + ct), tile_bytes);
        }
        cb_wait_front(cb_half, 1);                  // reader's rows are in the slot, its input read done
        cb_pop_front(cb_half, 1);
        cb_pop_front(cb_in, 32);
        for (uint32_t q = 0; q < QR; ++q) {
            noc_async_write(slot + q * tile_bytes, d.get_noc_addr(((Jt * 32 + q) * S1t + It) * Ct + ct), tile_bytes);
        }
        noc_async_write_barrier();
        if (i + 2 < num) {
            cb_reserve_back(cb_free, 1);
            cb_push_back(cb_free, 1);
        }
    }
}
