// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// permute_back_pk: x [C, N, N] channel-major -> out [N, N, H] channel-last, the re-tiling done by the packer.
// x may carry a batch [B, C, N, N] -> [B, N, N, H]: units run b slowest, so batch 1 reads the same pages in the same order.
// Per unit (b, ct, it, jt): the 32 channel tiles x[b][ct*32 + c][it, jt], in channel order, in groups of 8.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t ABL = get_compile_time_arg_val(1);   // bit 0: no DRAM read (diagnostic)
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    constexpr uint32_t ORD = get_compile_time_arg_val(3);   // 1: channel tile fastest (bank spread), 0: ct slowest
    constexpr uint32_t SW = get_compile_time_arg_val(4);    // 1: also write the odd 8-tile groups (permute_back_pk.SW)
    constexpr uint32_t Ht = get_compile_time_arg_val(5);
    constexpr uint32_t CO = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    const uint32_t step = get_arg_val<uint32_t>(2);     // 1, or the core count under round-robin (permute_back_pk.RR)
    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t SS = St * St;
    constexpr uint32_t CS = CT * SS;
    const uint32_t tb = get_tile_size(in_cb);
    constexpr uint32_t sw_cb = tt::CBIndex::c_17;
    // Unit u's odd groups (rows 8-15, 24-31), written once unit u + 1 is read so compute is never starved.
    auto write_odd = [&](uint32_t u) {
        const uint32_t b = u / CS, v = u % CS;
        const uint32_t ct = ORD ? v % CT : v / SS;
        const uint32_t cube = ORD ? v / CT : v % SS;
        const uint32_t ob = get_tile_size(sw_cb);
        const uint32_t page = ((b * St * 32 + (cube / St) * 32) * St + cube % St) * Ht + CO + ct;
        for (uint32_t g = 1; g < 4; g += 2) {
            uint32_t pg = page + g * 8 * St * Ht;
            cb_wait_front(sw_cb, 8);
            uint32_t src = get_read_ptr(sw_cb);
            for (uint32_t k = 0; k < 8; ++k, pg += St * Ht, src += ob) noc_async_write_page(pg, o, src);
            noc_async_writes_flushed();
            cb_pop_front(sw_cb, 8);
        }
    };

    for (uint32_t i = 0, u = u0; i < nunits; ++i, u += step) {
        const uint32_t b = u / CS, v = u % CS;
        const uint32_t ct = ORD ? v % CT : v / SS;
        const uint32_t cube = ORD ? v / CT : v % SS;          // it * St + jt
        uint32_t page = (b * CT + ct) * 32 * SS + cube;
        for (uint32_t g = 0; g < 4; ++g) {
            cb_reserve_back(in_cb, 8);
            uint32_t dst = get_write_ptr(in_cb);
            for (uint32_t c = 0; c < 8; ++c, page += SS, dst += tb)
                if constexpr (!(ABL & 1)) noc_async_read_page(page, x, dst);
            noc_async_read_barrier();
            cb_push_back(in_cb, 8);
        }
        if constexpr (SW) if (i) write_odd(u - step);
    }
    if constexpr (SW) {
        if (nunits) write_odd(u0 + (nunits - 1) * step);
        noc_async_write_barrier();
    }
}
