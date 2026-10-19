// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triatt_ln writer (tt_bio/triatt_ln.py): row tile r's NW output tiles, n = c * H + h for the four
// head-major chunks q, k, v, g and the last tile the [B, S, 32] pair bias (drain.hpp).
// H = 0 (apb_ln, triatt_ln.ln_proj): the NW tiles go row-major to one [B, S, NW * 32] tensor.
// A core of column group n0 (runtime arg 4) writes output tiles n0 .. n0 + NW - 1 of the NWF a row tile
// carries; the ones past NWF are padding and are dropped. SPLIT: only the even row ordinals, the reader
// drains the odd ones from c_4.
#include "drain.hpp"
#ifndef SPLIT
#define SPLIT 0
#endif

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

// The H = 0 build has one output accessor and must never instantiate the five-accessor drain, so write()
// picks the layout at compile time.
template <uint32_t H, uint32_t MT, uint32_t NW, uint32_t NWF>
void write(uint32_t r0, uint32_t nrows, uint32_t out_cb, uint32_t tb, uint32_t n0) {
    if constexpr (H == 0) {
        write_rows<NW, NWF>(r0, nrows, out_cb, tb, n0);
    } else {
        static_assert(NWF == 4 * H + 1);
        HeadDrain<H, MT, NW, 4, 0>(r0, nrows, out_cb, n0, 0, SPLIT ? 2 : 1).rest();
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
