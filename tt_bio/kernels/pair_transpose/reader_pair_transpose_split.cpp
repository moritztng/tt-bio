// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_transpose reader, split variant. Same units as reader_pair_transpose.cpp, but this core's
// reader also does the row shuffle for output tiles 0..QR-1 of each unit, which the plain variant
// leaves to the writer. The next unit's DRAM reads are issued before that shuffle so the two
// overlap. Two-slot scratch in c_16 (slot = unit parity); c_1 says "my rows of this unit are in
// the slot and I am done with its input", c_2 says "the writer has written slot k out".
//
// ZADD (pair_transpose_add) also reads the 32 z tiles each unit lands on into c_3, so that every DRAM
// read of the op is on this RISC's NoC and the writer's NoC carries only its DRAM writes.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, cb_half = 1, cb_free = 2, cb_out = 16;
    constexpr uint32_t S1t = get_compile_time_arg_val(0);
    constexpr uint32_t S2t = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr uint32_t ROW = get_compile_time_arg_val(3);   // bytes in one face row: 16 elements
    constexpr uint32_t QR = get_compile_time_arg_val(4);    // output tiles this kernel shuffles
    constexpr uint32_t ZADD = get_compile_time_arg_val(5);  // also read z into c_3
    constexpr uint32_t cb_z = 3;
    constexpr auto s_args = TensorAccessorArgs<6>();
    const auto s = TensorAccessor(s_args, src_addr);
    constexpr auto z_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(1));

    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t num = get_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_in);
    constexpr uint32_t FACE = 16 * ROW;
    const uint32_t out_base = get_write_ptr(cb_out);

    auto issue = [&](uint32_t u) {
        const uint32_t ct = u % Ct;
        const uint32_t It = (u / Ct) % S1t;
        const uint32_t Jt = u / (Ct * S1t);
        cb_reserve_back(cb_in, 32);
        uint32_t w = get_write_ptr(cb_in);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_read_page(((It * 32 + r) * S2t + Jt) * Ct + ct, s, w);
            w += tile_bytes;
        }
        return get_write_ptr(cb_in);
    };

    if (num == 0) {
        return;
    }
    uint32_t in = issue(first);
    noc_async_read_barrier();
    for (uint32_t i = 0; i < num; ++i) {
        cb_push_back(cb_in, 32);                    // the writer may start its rows of unit i
        uint32_t next = 0;
        if (i + 1 < num) {
            next = issue(first + i + 1);            // overlaps the shuffle below
        }
        if constexpr (ZADD) {
            const uint32_t u = first + i;
            const uint32_t ct = u % Ct;
            const uint32_t It = (u / Ct) % S1t;
            const uint32_t Jt = u / (Ct * S1t);
            cb_reserve_back(cb_z, 32);
            uint32_t lz = get_write_ptr(cb_z);
            for (uint32_t q = 0; q < 32; ++q, lz += tile_bytes) {
                noc_async_read_page(((Jt * 32 + q) * S1t + It) * Ct + ct, z, lz);
            }
        }
        if (i >= 2) {
            cb_wait_front(cb_free, 1);              // slot (i & 1) written out by the writer
            cb_pop_front(cb_free, 1);
        }
        const uint32_t slot = out_base + (i & 1) * 32 * tile_bytes;
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t src_tile = in + r * tile_bytes;
            const uint32_t dst_row = ((r >> 4) * 2) * FACE + (r & 15) * ROW;
            for (uint32_t q = 0; q < QR; ++q) {
                const uint32_t src = src_tile + ((q >> 4) * 2) * FACE + (q & 15) * ROW;
                const uint32_t dst = slot + q * tile_bytes + dst_row;
                noc_async_read(get_noc_addr(src), dst, ROW);
                noc_async_read(get_noc_addr(src + FACE), dst + FACE, ROW);
            }
        }
        noc_async_read_barrier();                   // this shuffle and the next unit's DRAM reads
        if constexpr (ZADD) {
            cb_push_back(cb_z, 32);
        }
        cb_reserve_back(cb_half, 1);
        cb_push_back(cb_half, 1);
        in = next;
    }
}
