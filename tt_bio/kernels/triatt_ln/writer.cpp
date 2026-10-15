// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triatt_ln writer (tt_bio/triatt_ln.py): row tile r's NW output tiles, n = c * H + h for the four
// head-major chunks q, k, v, g ([B, H, S, 32]: page ((r / MT) * H + h) * MT + r % MT, the
// HEAD_MAJOR_MT transform of kernels/triatt) and the last tile the [B, S, 32] pair bias (page r).
// H = 0 (apb_ln, triatt_ln.ln_proj): the NW tiles go row-major to one [B, S, NW * 32] tensor.
#include "api/dataflow/dataflow_api.h"

// Row tile of the i-th row this core writes. Plain: r0 + i. TR (runtime arg 2, reader.cpp): r0 is the first
// unit, 16 rows a unit, row k of unit (bt, jt, half) is row tile (bt * 32 + half * 16 + k) * MT + jt of z^T.
inline uint32_t row_of(uint32_t r0, uint32_t i) {
    if (!get_arg_val<uint32_t>(2)) return r0 + i;
    const uint32_t MT = get_arg_val<uint32_t>(3), u = r0 + i / 16;
    return ((u >> 1) / MT * 32 + (u & 1) * 16 + i % 16) * MT + (u >> 1) % MT;
}

// The H = 0 build has one output accessor and must never instantiate the five-accessor body. Outside a
// template `if constexpr` checks its untaken branch, and inside one a body whose accessor offsets do not
// depend on a template parameter is checked at definition, so write() picks the layout and write_heads
// takes its first offset CTA as a parameter.
template <uint32_t NW>
void write_rows(uint32_t r0, uint32_t nrows, uint32_t out_cb, uint32_t tb) {
    // apb_ln: row tile r's NW tiles are pages r * NW .. of the row-major [B, S, n] result.
    constexpr auto o_args = TensorAccessorArgs<3>();
    const auto oo = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    for (uint32_t i = 0; i < nrows; ++i) {
        const uint32_t r = row_of(r0, i);
        cb_wait_front(out_cb, NW);
        uint32_t src = get_read_ptr(out_cb);
        for (uint32_t j = 0; j < NW; ++j, src += tb) noc_async_write_page(r * NW + j, oo, src);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, NW);
    }
}

template <uint32_t H, uint32_t MT, uint32_t NW, uint32_t CTA>
void write_heads(uint32_t r0, uint32_t nrows, uint32_t out_cb, uint32_t tb) {
    static_assert(NW == 4 * H + 1);
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
    for (uint32_t i = 0; i < nrows; ++i) {
        const uint32_t r = row_of(r0, i);
        cb_wait_front(out_cb, NW);
        uint32_t src = get_read_ptr(out_cb);
        const uint32_t base = (r / MT) * H * MT + r % MT;
        for (uint32_t h = 0; h < H; ++h, src += tb) noc_async_write_page(base + h * MT, oq, src);
        for (uint32_t h = 0; h < H; ++h, src += tb) noc_async_write_page(base + h * MT, ok, src);
        for (uint32_t h = 0; h < H; ++h, src += tb) noc_async_write_page(base + h * MT, ov, src);
        for (uint32_t h = 0; h < H; ++h, src += tb) noc_async_write_page(base + h * MT, og, src);
        noc_async_write_page(r, ob, src);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, NW);
    }
}

template <uint32_t H, uint32_t MT, uint32_t NW>
void write(uint32_t r0, uint32_t nrows, uint32_t out_cb, uint32_t tb) {
    if constexpr (H == 0) {
        write_rows<NW>(r0, nrows, out_cb, tb);
    } else {
        write_heads<H, MT, NW, 3>(r0, nrows, out_cb, tb);
    }
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);    // head tiles per chunk; 0: plain [B, S, NW * 32] output
    constexpr uint32_t MT = get_compile_time_arg_val(1);   // row tiles per batch
    constexpr uint32_t NW = get_compile_time_arg_val(2);   // output tiles per row tile
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    const uint32_t tb = get_tile_size(out_cb);
    write<H, MT, NW>(r0, nrows, out_cb, tb);
    noc_async_write_barrier();
}
