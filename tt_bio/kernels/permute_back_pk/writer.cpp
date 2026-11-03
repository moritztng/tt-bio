// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// permute_back_pk: unit (b, ct, it, jt)'s 32 output tiles, row i = it*32 + i of out[b] [N, N, H] at column tile jt and
// channel tile CO + ct, written whole, WB per flush.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t CO = get_compile_time_arg_val(2);
    constexpr uint32_t ABL = get_compile_time_arg_val(3);   // bit 2: no DRAM write (diagnostic). Wrong output, timing only:
                                                            // bit 16 writes the unit to 32 consecutive pages, bit 32 adds
                                                            // k to the k-th page so a unit's writes rotate across banks
    constexpr uint32_t WB = get_compile_time_arg_val(4);    // tiles per flush
    constexpr uint32_t CT = get_compile_time_arg_val(5);
    constexpr uint32_t ORD = get_compile_time_arg_val(6);   // as the reader's
    constexpr auto o_args = TensorAccessorArgs<7>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    const uint32_t step = get_arg_val<uint32_t>(2);     // 1, or the core count under round-robin (permute_back_pk.RR)
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t SS = St * St;
    constexpr uint32_t CS = CT * SS;
    const uint32_t tb = get_tile_size(out_cb);

    for (uint32_t i = 0, u = u0; i < nunits; ++i, u += step) {
        const uint32_t b = u / CS, v = u % CS;
        const uint32_t ct = ORD ? v % CT : v / SS;
        const uint32_t cube = ORD ? v / CT : v % SS;
        const uint32_t it = cube / St;
        const uint32_t jt = cube % St;
        uint32_t page = ((b * St * 32 + it * 32) * St + jt) * Ht + CO + ct;
        if constexpr (ABL & 16) page = u * 32;
        constexpr uint32_t pstride = (ABL & 16) ? 1 : (ABL & 32) ? St * Ht + 1 : St * Ht;
        for (uint32_t i0 = 0; i0 < 32; i0 += WB) {
            cb_wait_front(out_cb, WB);
            uint32_t src = get_read_ptr(out_cb);
            for (uint32_t k = 0; k < WB; ++k, page += pstride, src += tb)
                if constexpr (!(ABL & 4)) noc_async_write_page((ABL & 32) ? page % (St * 32 * St * Ht) : page, o, src);
            noc_async_writes_flushed();
            cb_pop_front(out_cb, WB);
        }
    }
    noc_async_write_barrier();
}
