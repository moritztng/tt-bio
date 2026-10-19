// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triatt_ln writer (tt_bio/triatt_ln.py): row tile r's NW output tiles, n = c * H + h for the four
// head-major chunks q, k, v, g ([B, H, S, 32]: page ((r / MT) * H + h) * MT + r % MT, the
// HEAD_MAJOR_MT transform of kernels/triatt) and the last tile the [B, S, 32] pair bias (page r).
// H = 0 (apb_ln, triatt_ln.ln_proj): the NW tiles go row-major to one [B, S, NW * 32] tensor.
// A core of column group n0 (runtime arg 4) writes output tiles n0 .. n0 + NW - 1 of the NWF a row tile
// carries; the ones past NWF are padding and are dropped.
#include "api/dataflow/dataflow_api.h"

// Row tile of the i-th row this core writes. Plain: r0 + i. TR (runtime arg 2, reader.cpp): r0 is the first
// unit, 16 rows a unit, row k of unit (bt, jt, half) is row tile (bt * 32 + half * 16 + k) * MT + jt of z^T.
inline uint32_t row_of(uint32_t r0, uint32_t i) {
    if (!get_arg_val<uint32_t>(2)) return r0 + i;
    const uint32_t MT = get_arg_val<uint32_t>(3), u = r0 + i / 16;
    return ((u >> 1) / MT * 32 + (u & 1) * 16 + i % 16) * MT + (u >> 1) % MT;
}

#ifndef ABL_W
#define ABL_W 0   // bench-only (triatt_ln.ABL bit 4): skip the DRAM writes
#endif
#ifndef NQ
#define NQ 0      // triatt_ln_b8: the first NQ = 3H tiles (q, k, v) come bfp8 from c_2, the rest bf16 from c_4
#endif

#ifndef ABL_LIN
#define ABL_LIN 0 // bench-only (triatt_ln.ABL bit 16): q's page count; every tile goes to q page (r * NWF + n) mod it
#endif

// The H = 0 build has one output accessor and must never instantiate the five-accessor body. Outside a
// template `if constexpr` checks its untaken branch, and inside one a body whose accessor offsets do not
// depend on a template parameter is checked at definition, so write() picks the layout and write_heads
// takes its first offset CTA as a parameter.
template <uint32_t NW, uint32_t NWF>
void write_rows(uint32_t r0, uint32_t nrows, uint32_t out_cb, uint32_t tb, uint32_t n0) {
    // apb_ln: row tile r's NW tiles are pages r * NW .. of the row-major [B, S, n] result.
    constexpr auto o_args = TensorAccessorArgs<4>();
    const auto oo = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    for (uint32_t i = 0; i < nrows; ++i) {
        const uint32_t r = row_of(r0, i);
        cb_wait_front(out_cb, NW);
        uint32_t src = get_read_ptr(out_cb);
        for (uint32_t j = 0; j < NW && n0 + j < NWF; ++j, src += tb) if constexpr (!ABL_W) noc_async_write_page(r * NWF + n0 + j, oo, src);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, NW);
    }
}

template <uint32_t H, uint32_t MT, uint32_t NW, uint32_t CTA>
void write_heads(uint32_t r0, uint32_t nrows, uint32_t out_cb, uint32_t tb, uint32_t n0) {
    constexpr auto q_args = TensorAccessorArgs<CTA>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto oq = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0));
    const auto ok = TensorAccessor(k_args, get_common_arg_val<uint32_t>(1));
    const auto ov = TensorAccessor(v_args, get_common_arg_val<uint32_t>(2));
    const auto og = TensorAccessor(g_args, get_common_arg_val<uint32_t>(3));
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(4));
    constexpr uint32_t hi_cb = NQ ? tt::CBIndex::c_4 : out_cb;   // g, bias (NQ) or nothing
    const uint32_t tb_hi = get_tile_size(hi_cb);
    for (uint32_t i = 0; i < nrows; ++i) {
        const uint32_t r = row_of(r0, i);
        cb_wait_front(out_cb, NQ ? NQ : NW);
        if constexpr (NQ) cb_wait_front(hi_cb, NW - NQ);
        uint32_t src = get_read_ptr(out_cb);
        const uint32_t base = (r / MT) * H * MT + r % MT;
        for (uint32_t j = 0, n = n0; j < NW && n <= 4 * H; ++j, ++n) {
            if (NQ && j == NQ) src = get_read_ptr(hi_cb);
            const uint32_t page = base + (n % H) * MT;
            const uint32_t s = src;
            src += (NQ && j >= NQ) ? tb_hi : tb;
            if constexpr (ABL_W) continue;
            if constexpr (ABL_LIN) { noc_async_write_page((r * (4 * H + 1) + n) % (ABL_LIN ? ABL_LIN : 1), oq, s); continue; }
            switch (n / H) {
                case 0: noc_async_write_page(page, oq, s); break;
                case 1: noc_async_write_page(page, ok, s); break;
                case 2: noc_async_write_page(page, ov, s); break;
                case 3: noc_async_write_page(page, og, s); break;
                default: noc_async_write_page(r, ob, s);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(out_cb, NQ ? NQ : NW);
        if constexpr (NQ) cb_pop_front(hi_cb, NW - NQ);
    }
}

template <uint32_t H, uint32_t MT, uint32_t NW, uint32_t NWF>
void write(uint32_t r0, uint32_t nrows, uint32_t out_cb, uint32_t tb, uint32_t n0) {
    if constexpr (H == 0) {
        write_rows<NW, NWF>(r0, nrows, out_cb, tb, n0);
    } else {
        static_assert(NWF == 4 * H + 1);
        write_heads<H, MT, NW, 4>(r0, nrows, out_cb, tb, n0);
    }
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);    // head tiles per chunk; 0: plain [B, S, NW * 32] output
    constexpr uint32_t MT = get_compile_time_arg_val(1);   // row tiles per batch
    constexpr uint32_t NW = get_compile_time_arg_val(2);   // output tiles per row tile, this group
    constexpr uint32_t NWF = get_compile_time_arg_val(3);  // output tiles per row tile, all groups
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    const uint32_t tb = get_tile_size(out_cb);
    write<H, MT, NW, NWF>(r0, nrows, out_cb, tb, get_arg_val<uint32_t>(4));
    noc_async_write_barrier();
}
