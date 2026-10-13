// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul input side T1 writer, in the order compute produces: per unit (b, xt, yt) first the 32 row
// tiles of g_out (NTG tiles each, z's own [.., S, S, N] layout, one page write per tile), then
// gin_moved's writer as is: per channel tile q of [a | b], 32 tiles in il order, gathered face row by
// face row into the [B, C, S, S] tiles of a (q < CT) and b. NEVER RUN ON A DEVICE YET.
// ABL (bench-only define, trimul_tail.T1_ABL; 0 in every model path): bits that skip one stage's
// work while keeping its circular-buffer traffic, to time each stage by difference (perf/spd_trimul/in_t1.py --abl).
//   1 layer_norm, 2 g_out matmul, 4 [a | b] matmuls, 8 gate multiply, 16 z reads, 32 writer gather, 64 writer writes.
#ifndef ABL
#define ABL 0
#endif
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CT2 = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);    // output tiles gathered per read barrier
    constexpr uint32_t NTG = get_compile_time_arg_val(3);   // g_out channel tiles, 0: none
    constexpr uint32_t R = get_compile_time_arg_val(4);     // row tiles per item: 32 (unit) or 16 (half)
    constexpr auto a_args = TensorAccessorArgs<5>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto oa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0));
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1));
    const auto og = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t go_cb = tt::CBIndex::c_19;
    constexpr uint32_t stage_cb = tt::CBIndex::c_24;
    constexpr uint32_t CT = CT2 / 2;
    constexpr uint32_t SS = St * St;
    constexpr uint32_t S = St * 32;
    constexpr uint32_t TB = 2048, FB = 512, RB = 32;
    static_assert(R == 32 || R == 16, "a half item is the face pair 0-1 or 2-3 of every tile");
    constexpr uint32_t NH = 32 / R;
    // A half item writes only its face pair: 1024 contiguous bytes at offset h * 1024 of the tile.
    constexpr uint32_t WB = R * 64;

    cb_reserve_back(stage_cb, 2 * NB);
    const uint32_t stage0 = get_write_ptr(stage_cb);
    noc_async_read_one_packet_set_state(get_noc_addr(stage0), RB);
    bool dirty[2] = {false, false};

    for (uint32_t v = u0; v < u0 + nunits; ++v) {
        const uint32_t u = v / NH;
        const uint32_t h = v % NH;               // 0, or 1 for rows 16-31 of a half item
        const uint32_t b = u / SS;
        const uint32_t cube = u % SS;          // xt * St + yt
        const uint32_t xt = cube / St;
        const uint32_t yt = cube % St;
        if constexpr (NTG) {
            // trimul_tail.t1_g_page: row tile (b, xt*32 + il, yt), channel tile n.
            uint32_t page = ((b * S + xt * 32 + h * R) * St + yt) * NTG;
            for (uint32_t il = 0; il < R; ++il, page += St * NTG) {
                cb_wait_front(go_cb, NTG);
                uint32_t r = get_read_ptr(go_cb);
                for (uint32_t n = 0; n < NTG; ++n, r += TB)
                    if constexpr (!(ABL & 64)) noc_async_write_page(page + n, og, r);
                noc_async_writes_flushed();
                cb_pop_front(go_cb, NTG);
            }
        }
        for (uint32_t q = 0; q < CT2; ++q) {
            const bool is_a = q < CT;
            uint32_t page = (b * CT + (is_a ? q : q - CT)) * 32 * SS + cube;
            cb_wait_front(out_cb, R);
            const uint32_t src = get_read_ptr(out_cb);
            for (uint32_t c0 = 0; c0 < 32; c0 += NB) {
                const uint32_t grp = (c0 / NB) & 1u;
                const uint32_t gbase = stage0 + grp * NB * TB;
                if (dirty[grp]) noc_async_writes_flushed();
                for (uint32_t j = 0; j < NB && !(ABL & 32); ++j) {
                    const uint32_t c = c0 + j;
                    const uint32_t st = gbase + j * TB;
                    const uint32_t sc = src + (c >> 4) * 2 * FB + (c & 15) * RB;
                    // Source tile il of the item holds tile row h * R + il; a half item stages its
                    // face pair at the front of the stage tile (faces 0-1 layout either way).
                    uint32_t s0 = sc, s1 = sc + FB;
                    for (uint32_t f = 0; f < R / 16; ++f) {
                        uint32_t d0 = st + 2 * f * FB, d1 = d0 + FB;
                        for (uint32_t il = 0; il < 16; ++il, s0 += TB, s1 += TB, d0 += RB, d1 += RB) {
                            noc_async_read_one_packet_with_state(s0, d0);
                            noc_async_read_one_packet_with_state(s1, d1);
                        }
                    }
                }
                noc_async_read_barrier();
                for (uint32_t j = 0; j < NB; ++j, page += SS)
                    if constexpr (!(ABL & 64)) noc_async_write(gbase + j * TB, is_a ? oa.get_noc_addr(page, h * WB) : ob.get_noc_addr(page, h * WB), WB);
                dirty[grp] = true;
            }
            noc_async_write_barrier();
            dirty[0] = dirty[1] = false;
            cb_pop_front(out_cb, R);
        }
    }
    noc_async_write_barrier();
}
