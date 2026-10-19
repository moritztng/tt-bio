// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triatt_ln's output drain, shared by writer.cpp and (SPLIT) reader.cpp. With SPLIT compute packs even row
// ordinals to c_2 for the writer and odd ones to c_4 for the reader, so the 4x-larger write half leaves on both
// NOCs, each RISC on its own in the dedicated mode. One NOC drains ~120 GB/s into Wormhole DRAM (wbw2: every
// stock op writes at 100-118 GB/s) and the dynamic mode that lets one RISC use both costs more than it frees.
#pragma once
#include "api/dataflow/dataflow_api.h"

#ifndef ABL_W
#define ABL_W 0   // bench-only (triatt_ln.ABL bit 4): skip the DRAM writes
#endif
#ifndef ABL_LIN
#define ABL_LIN 0 // bench-only (triatt_ln.ABL bit 16): q's page count; every tile goes to q page (r * NWF + n) mod it
#endif

// Row tile of the i-th row this core writes. Plain: r0 + i. TR (runtime arg 2, reader.cpp): r0 is the first
// unit, 16 rows a unit, row k of unit (bt, jt, half) is row tile (bt * 32 + half * 16 + k) * MT + jt of z^T.
inline uint32_t row_of(uint32_t r0, uint32_t i) {
    if (!get_arg_val<uint32_t>(2)) return r0 + i;
    const uint32_t MT = get_arg_val<uint32_t>(3), u = r0 + i / 16;
    return ((u >> 1) / MT * 32 + (u & 1) * 16 + i % 16) * MT + (u >> 1) % MT;
}

// Row tile r's NW tiles at src to the head-major q, k, v, g ([B, H, S, 32]: page ((r / MT) * H + h) * MT + r % MT,
// the HEAD_MAJOR_MT transform of kernels/triatt) and the [B, S, 32] pair bias (page r). n0: first tile of this
// core's column group; tiles past 4H are padding and dropped.
template <uint32_t H, uint32_t MT, uint32_t NW, typename Q, typename K, typename V, typename G, typename B>
inline void write_head_row(const Q& oq, const K& ok, const V& ov, const G& og, const B& ob, uint32_t r, uint32_t src,
                           uint32_t tb, uint32_t n0) {
    const uint32_t base = (r / MT) * H * MT + r % MT;
    for (uint32_t j = 0, n = n0; j < NW && n <= 4 * H; ++j, ++n, src += tb) {
        const uint32_t page = base + (n % H) * MT;
        if constexpr (ABL_W) continue;
        if constexpr (ABL_LIN) { noc_async_write_page((r * (4 * H + 1) + n) % (ABL_LIN ? ABL_LIN : 1), oq, src); continue; }
        switch (n / H) {
            case 0: noc_async_write_page(page, oq, src); break;
            case 1: noc_async_write_page(page, ok, src); break;
            case 2: noc_async_write_page(page, ov, src); break;
            case 3: noc_async_write_page(page, og, src); break;
            default: noc_async_write_page(r, ob, src);
        }
    }
}

// The five output accessors from compile-time args at CTA and common runtime args CA .. CA + 4, and the rows
// i = first, first + step, ... of this core drained from cb.
template <uint32_t H, uint32_t MT, uint32_t NW, uint32_t CTA, uint32_t CA>
struct HeadDrain {
    static constexpr auto q_args = TensorAccessorArgs<CTA>();
    static constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    static constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    static constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    static constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    decltype(TensorAccessor(q_args, 0)) oq;
    decltype(TensorAccessor(k_args, 0)) ok;
    decltype(TensorAccessor(v_args, 0)) ov;
    decltype(TensorAccessor(g_args, 0)) og;
    decltype(TensorAccessor(b_args, 0)) ob;
    uint32_t r0, rows, cb, tb, n0, i, step;
    HeadDrain(uint32_t r0_, uint32_t rows_, uint32_t cb_, uint32_t n0_, uint32_t first, uint32_t step_)
        : oq(TensorAccessor(q_args, get_common_arg_val<uint32_t>(CA))),
          ok(TensorAccessor(k_args, get_common_arg_val<uint32_t>(CA + 1))),
          ov(TensorAccessor(v_args, get_common_arg_val<uint32_t>(CA + 2))),
          og(TensorAccessor(g_args, get_common_arg_val<uint32_t>(CA + 3))),
          ob(TensorAccessor(b_args, get_common_arg_val<uint32_t>(CA + 4))),
          r0(r0_), rows(rows_), cb(cb_), tb(get_tile_size(cb_)), n0(n0_), i(first), step(step_) {}
    void one() {
        cb_wait_front(cb, NW);
        write_head_row<H, MT, NW>(oq, ok, ov, og, ob, row_of(r0, i), get_read_ptr(cb), tb, n0);
        noc_async_writes_flushed();
        cb_pop_front(cb, NW);
        i += step;
    }
    // Rows already packed, without blocking (the reader calls this between its reads).
    void ready() { while (i < rows && cb_pages_available_at_front(cb, NW)) one(); }
    void rest() { while (i < rows) one(); }
};
