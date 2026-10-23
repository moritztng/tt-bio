// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul input side T1 writer, in the order compute produces: per unit (b, xt, yt) first the 32 row
// tiles of g_out (NTG tiles each, z's own [.., S, S, N] layout, one page write per tile), then
// gin_moved's writer as is: per channel tile q of [a | b], 32 tiles in il order, gathered face row by
// face row into the [B, C, S, S] tiles of a (q < CT) and b. One launch writes the channel group reader.cpp
// names; with IL < 32 a unit holds rows h * IL .. of an x tile: IL = 16 writes that half's two faces (1 KB),
// IL = 8 the unit's 8 rows of two faces (two 256 B writes).
// ABL (bench-only define, trimul_tail.T1_ABL; 0 in every model path): bits that skip one stage's
// work while keeping its circular-buffer traffic, to time each stage by difference (perf/spd_trimul/in_t1.py --abl).
//   1 layer_norm, 2 g_out matmul, 4 [a | b] matmuls, 8 gate multiply, 16 z reads, 32 writer gather, 64 writer writes.
// T1B8 (lever trimul_t1b8): a and b leave compute as bfp8 tiles. bfp8 keeps one exponent per 16-value face row,
// the unit this gather moves, so the re-tiling is lossless: 16 mantissa bytes per face row by NOC as before, and
// the face row's exponent byte by a core store (the NOC cannot move a single unaligned byte). At IL 8 the items
// come in face-pair pairs (unit.hpp) and the second writes both items' exponents (gather.hpp).
// T1SPLIT (trimul_tail.T1_SPLIT): compute sends the launch's even channel tiles to reader.cpp (c_21), which gathers
// and writes them; this writer takes the odd ones and g_out. The gather is gather.hpp on both cores.
#ifndef ABL
#define ABL 0
#endif
#ifndef T1B8
#define T1B8 0
#endif
// T1XOUT (trimul_tail.T1_LN1): this launch also stores xn, which compute sends first in each unit, row tile
// by row tile in z's own layout, for the later channel-group launches to read instead of renormalising z.
#ifndef T1XOUT
#define T1XOUT 0
#endif
#ifndef T1SPLIT
#define T1SPLIT 0
#endif
#include "api/dataflow/dataflow_api.h"
#include "gather.hpp"
#include "unit.hpp"

void kernel_main() {
    constexpr uint32_t CT2 = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);    // output tiles gathered per read barrier
    constexpr uint32_t NG = get_compile_time_arg_val(3);    // g_out channel tiles of this launch, 0: none
    constexpr uint32_t Q0 = get_compile_time_arg_val(4);
    constexpr uint32_t QN = get_compile_time_arg_val(5);
    constexpr uint32_t N0 = get_compile_time_arg_val(6);
    constexpr uint32_t NTGT = get_compile_time_arg_val(7);  // g_out channel tiles in all
    constexpr uint32_t IL = get_compile_time_arg_val(8);    // row tiles per unit, 32, 16 or 8
    constexpr uint32_t H = 32 / IL;
    constexpr auto a_args = TensorAccessorArgs<9>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto oa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0));
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1));
    const auto og = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2));
#if T1XOUT
    constexpr uint32_t Kt = get_compile_time_arg_val(g_args.next_compile_time_args_offset());
    constexpr auto xn_args = TensorAccessorArgs<g_args.next_compile_time_args_offset() + 1>();
    const auto oxn = TensorAccessor(xn_args, get_common_arg_val<uint32_t>(3));
    constexpr uint32_t xo_cb = tt::CBIndex::c_20;
#endif
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t go_cb = tt::CBIndex::c_19;
    constexpr uint32_t stage_cb = tt::CBIndex::c_24;
    constexpr uint32_t CT = CT2 / 2;
    constexpr uint32_t SS = St * St;
    constexpr uint32_t S = St * 32;
    constexpr uint32_t TB = t1_gather::TB;
    t1_gather::Gather<NB, IL, ABL, T1B8> gather;
    gather.init(stage_cb, tt::CBIndex::c_26);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const auto [b, xt, h, yt] = t1_unit::decode<St, H, T1B8 && IL == 8>(u);
        const uint32_t cube = xt * St + yt;
#if T1XOUT
        {
            uint32_t page = ((b * S + xt * 32 + h * IL) * St + yt) * Kt;   // z's row tile pages
            for (uint32_t il = 0; il < IL; ++il, page += St * Kt) {
                cb_wait_front(xo_cb, Kt);
                uint32_t r = get_read_ptr(xo_cb);
                for (uint32_t dt = 0; dt < Kt; ++dt, r += TB) noc_async_write_page(page + dt, oxn, r);
                noc_async_writes_flushed();
                cb_pop_front(xo_cb, Kt);
            }
        }
#endif
        if constexpr (NG) {
            // trimul_tail.t1_g_page: row tile (b, xt*32 + il, yt), channel tile n.
            uint32_t page = ((b * S + xt * 32 + h * IL) * St + yt) * NTGT + N0;
            const uint32_t tg = get_tile_size(go_cb);    // bf16, or bfp8 under trimul_t1g8
            for (uint32_t il = 0; il < IL; ++il, page += St * NTGT) {
                cb_wait_front(go_cb, NG);
                uint32_t r = get_read_ptr(go_cb);
                for (uint32_t n = 0; n < NG; ++n, r += tg)
                    if constexpr (!(ABL & 64)) noc_async_write_page(page + n, og, r);
                noc_async_writes_flushed();
                cb_pop_front(go_cb, NG);
            }
        }
        for (uint32_t q = Q0 + T1SPLIT; q < Q0 + QN; q += 1 + T1SPLIT) {
            const uint32_t page = (b * CT + (q < CT ? q : q - CT)) * 32 * SS + cube;
            const uint32_t slot = (q - Q0) / (1 + T1SPLIT);
            if (q < CT) gather.tile(out_cb, oa, page, h * IL, SS, slot);
            else gather.tile(out_cb, ob, page, h * IL, SS, slot);
        }
    }
    noc_async_write_barrier();
}
