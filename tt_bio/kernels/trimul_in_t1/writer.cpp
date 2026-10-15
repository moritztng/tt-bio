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
//   128 / 256 (compute only): the [a | b] matmul_block calls / the gate sigmoid alone.
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
    constexpr uint32_t NG = get_compile_time_arg_val(3);    // g_out channel tiles of this launch, 0: none
    constexpr uint32_t Q0 = get_compile_time_arg_val(4);
    constexpr uint32_t QN = get_compile_time_arg_val(5);
    constexpr uint32_t N0 = get_compile_time_arg_val(6);
    constexpr uint32_t NTGT = get_compile_time_arg_val(7);  // g_out channel tiles in all
    constexpr uint32_t IL = get_compile_time_arg_val(8);    // row tiles per unit, 32, 16 or 8
    constexpr uint32_t H = 32 / IL;
    static_assert(!T1B8 || IL == 32, "the bfp8 gather moves whole tiles");
    constexpr auto a_args = TensorAccessorArgs<9>();
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
        const uint32_t b = u / (SS * H);
        const uint32_t xt = (u / (St * H)) % St;
        const uint32_t h = (u / St) % H;
        const uint32_t yt = u % St;
        const uint32_t cube = xt * St + yt;
        // The unit's rows r0 .. r0 + IL - 1 of each output tile: the staged tile keeps them where the output
        // has them, so one write (IL >= 16) or one per face pair column (IL = 8) moves them.
        const uint32_t r0 = h * IL;
        const uint32_t off0 = IL == 32 ? 0 : (r0 >> 4) * 2 * FB + (r0 & 15) * RB;
        const uint32_t wlen = IL == 32 ? TB : (IL == 16 ? 2 * FB : IL * RB);
        if constexpr (NG) {
            // trimul_tail.t1_g_page: row tile (b, xt*32 + il, yt), channel tile n.
            uint32_t page = ((b * S + xt * 32 + h * IL) * St + yt) * NTGT + N0;
            for (uint32_t il = 0; il < IL; ++il, page += St * NTGT) {
                cb_wait_front(go_cb, NG);
                uint32_t r = get_read_ptr(go_cb);
                for (uint32_t n = 0; n < NG; ++n, r += TB)
                    if constexpr (!(ABL & 64)) noc_async_write_page(page + n, og, r);
                noc_async_writes_flushed();
                cb_pop_front(go_cb, NG);
            }
        }
        for (uint32_t q = Q0; q < Q0 + QN; ++q) {
            const bool is_a = q < CT;
            uint32_t page = (b * CT + (is_a ? q : q - CT)) * 32 * SS + cube;
            cb_wait_front(out_cb, IL);
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
                    // source row c of tile il: face (c >> 4) * 2 (+1 for the right half), face row c & 15;
                    // destination row r0 + il, the same face arithmetic
                    uint32_t s0 = src + (c >> 4) * 2 * FB + (c & 15) * RB;
                    for (uint32_t il = 0; il < IL; ++il, s0 += TB) {
                        const uint32_t r = r0 + il;
                        const uint32_t d0 = st + (r >> 4) * 2 * FB + (r & 15) * RB;
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s0 + FB, d0 + FB);
                    }
                }
#endif
                noc_async_read_barrier();
                for (uint32_t j = 0; j < NB; ++j, page += SS) {
                    if constexpr (ABL & 64) continue;
                    const uint32_t st = gbase + j * TB;
                    if constexpr (T1B8) {
                        noc_async_write(st, is_a ? oa.get_noc_addr(page) : ob.get_noc_addr(page), TB8);
                    } else {
                        const uint64_t dst = is_a ? oa.get_noc_addr(page, off0) : ob.get_noc_addr(page, off0);
                        noc_async_write(st + off0, dst, wlen);
                        if constexpr (IL == 8) noc_async_write(st + off0 + FB, dst + FB, wlen);
                    }
                }
                dirty[grp] = true;
            }
            noc_async_write_barrier();
            dirty[0] = dirty[1] = false;
            cb_pop_front(out_cb, IL);
        }
    }
    noc_async_write_barrier();
}
