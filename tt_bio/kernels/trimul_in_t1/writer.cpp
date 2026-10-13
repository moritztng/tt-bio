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
// T1B8 (lever trimul_t1b8): a and b leave compute as bfp8 tiles. bfp8 keeps one exponent per 16-value face row,
// the unit this gather moves, so the re-tiling is lossless: 16 mantissa bytes per face row by NOC as before, and
// the face row's exponent byte by a core store (64 per tile; the NOC cannot move a single unaligned byte).
#ifndef ABL
#define ABL 0
#endif
#ifndef T1B8
#define T1B8 0
#endif
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CT2 = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);    // output tiles gathered per read barrier
    constexpr uint32_t NTG = get_compile_time_arg_val(3);   // g_out channel tiles, 0: none
    constexpr auto a_args = TensorAccessorArgs<4>();
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
    // bfp8_b tile: 64 exponent bytes (face-major, one per face row), then 4 faces x 16 rows x 16 mantissa bytes.
    constexpr uint32_t TB8 = 1088, EB = 64, FB8 = 256, RB8 = 16;

    cb_reserve_back(stage_cb, 2 * NB);
    const uint32_t stage0 = get_write_ptr(stage_cb);
    noc_async_read_one_packet_set_state(get_noc_addr(stage0), T1B8 ? RB8 : RB);
    bool dirty[2] = {false, false};

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / SS;
        const uint32_t cube = u % SS;          // xt * St + yt
        const uint32_t xt = cube / St;
        const uint32_t yt = cube % St;
        if constexpr (NTG) {
            // trimul_tail.t1_g_page: row tile (b, xt*32 + il, yt), channel tile n.
            uint32_t page = ((b * S + xt * 32) * St + yt) * NTG;
            for (uint32_t il = 0; il < 32; ++il, page += St * NTG) {
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
            cb_wait_front(out_cb, 32);
            const uint32_t src = get_read_ptr(out_cb);
            for (uint32_t c0 = 0; c0 < 32; c0 += NB) {
                const uint32_t grp = (c0 / NB) & 1u;
                const uint32_t gbase = stage0 + grp * NB * TB;
                if (dirty[grp]) noc_async_writes_flushed();
#if T1B8
                for (uint32_t j = 0; j < NB && !(ABL & 32); ++j) {
                    const uint32_t c = c0 + j;
                    const uint32_t st = gbase + j * TB;
                    // source row c of tile il: face (c >> 4) * 2 (+1 for the right half), face row c & 15
                    const uint32_t sf = (c >> 4) * 2, sr = c & 15;
                    volatile tt_l1_ptr uint8_t* se = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
                    volatile tt_l1_ptr uint8_t* de = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(st);
                    for (uint32_t il = 0; il < 32; ++il) {
                        const uint32_t df = (il >> 4) * 2, dr = il & 15;
                        const uint32_t s0 = src + il * TB8 + EB + sf * FB8 + sr * RB8;
                        const uint32_t d0 = st + EB + df * FB8 + dr * RB8;
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s0 + FB8, d0 + FB8);
                        de[df * 16 + dr] = se[il * TB8 + sf * 16 + sr];
                        de[(df + 1) * 16 + dr] = se[il * TB8 + (sf + 1) * 16 + sr];
                    }
                }
#else
                for (uint32_t j = 0; j < NB && !(ABL & 32); ++j) {
                    const uint32_t c = c0 + j;
                    const uint32_t st = gbase + j * TB;
                    const uint32_t sc = src + (c >> 4) * 2 * FB + (c & 15) * RB;
                    uint32_t s0 = sc, s1 = sc + FB, d0 = st, d1 = st + FB;
                    for (uint32_t il = 0; il < 16; ++il, s0 += TB, s1 += TB, d0 += RB, d1 += RB) {
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s1, d1);
                    }
                    d0 = st + 2 * FB;
                    d1 = st + 3 * FB;
                    for (uint32_t il = 0; il < 16; ++il, s0 += TB, s1 += TB, d0 += RB, d1 += RB) {
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s1, d1);
                    }
                }
#endif
                noc_async_read_barrier();
                for (uint32_t j = 0; j < NB; ++j, page += SS)
                    if constexpr (!(ABL & 64)) noc_async_write(gbase + j * TB, is_a ? oa.get_noc_addr(page) : ob.get_noc_addr(page), T1B8 ? TB8 : TB);
                dirty[grp] = true;
            }
            noc_async_write_barrier();
            dirty[0] = dirty[1] = false;
            cb_pop_front(out_cb, 32);
        }
    }
    noc_async_write_barrier();
}
