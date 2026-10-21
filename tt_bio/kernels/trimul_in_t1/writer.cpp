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
//   1 layer_norm, 2 g_out matmul, 4 [a | b] matmuls, 8 gate multiply, 16 z reads, 32 writer gather, 64 writer writes,
//   128 the bfp8 gather's exponent byte stores.
// T1B8 (lever trimul_t1b8): a and b leave compute as bfp8 tiles. bfp8 keeps one exponent per 16-value face row,
// the unit this gather moves, so the re-tiling is lossless: 16 mantissa bytes per face row by NOC as before, and
// the face row's exponent byte by a core store (64 per tile; the NOC cannot move a single unaligned byte).
// With IL = 16 a unit's rows are two whole faces of each output tile, whose 32 exponent bytes and 512
// mantissa bytes are each contiguous in the bfp8 tile: two writes per tile, no byte shared with the other half.
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
    static_assert(!T1B8 || IL >= 16, "the bfp8 gather moves whole faces");
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
                // The NB channels c0 .. c0 + NB - 1 are consecutive face rows of one source face, so their
                // exponent bytes are adjacent: one aligned word load per source tile and face. Four consecutive
                // rows of a destination tile have adjacent exponent bytes too, so they are stored as one word:
                // byte stores to L1 cost ~2.3 ms of a 9 ms call at 512 tokens on Wormhole.
                static_assert(NB == 1 || NB == 2 || NB == 4, "exponent bytes of a barrier group share one word");
                static_assert(IL % 4 == 0, "four rows per exponent word");
                if constexpr (!(ABL & 32)) {
                    const uint32_t sf = (c0 >> 4) * 2, sr0 = c0 & 15;
                    for (uint32_t i4 = 0; i4 < IL; i4 += 4) {
                        uint32_t e0[NB], e1[NB];
                        for (uint32_t j = 0; j < NB; ++j) e0[j] = e1[j] = 0;
                        for (uint32_t k = 0; k < 4; ++k) {
                            const uint32_t il = i4 + k, r = r0 + il;
                            const uint32_t df = (r >> 4) * 2, dr = r & 15;
                            const uint32_t ea = src + il * TB8 + sf * 16 + sr0;
                            const uint32_t sh = (ea & 3) * 8;
                            const uint32_t w0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ea & ~3u) >> sh;
                            const uint32_t w1 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>((ea + 16) & ~3u) >> sh;
                            for (uint32_t j = 0; j < NB; ++j) {
                                // source row c0 + j of tile il: face sf (+1 for the right half), face row sr0 + j
                                const uint32_t s0 = src + il * TB8 + EB + sf * FB8 + (sr0 + j) * RB8;
                                const uint32_t d0 = gbase + j * TB + EB + df * FB8 + dr * RB8;
                                noc_async_read_one_packet_with_state(s0, d0);
                                noc_async_read_one_packet_with_state(s0 + FB8, d0 + FB8);
                                e0[j] |= ((w0 >> (8 * j)) & 0xff) << (8 * k);
                                e1[j] |= ((w1 >> (8 * j)) & 0xff) << (8 * k);
                            }
                        }
                        if constexpr (ABL & 128) continue;
                        const uint32_t r = r0 + i4, df = (r >> 4) * 2, dr = r & 15;
                        for (uint32_t j = 0; j < NB; ++j) {
                            const uint32_t st = gbase + j * TB;
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st + df * 16 + dr) = e0[j];
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st + (df + 1) * 16 + dr) = e1[j];
                        }
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
                    if constexpr (T1B8 && IL == 32) {
                        noc_async_write(st, is_a ? oa.get_noc_addr(page) : ob.get_noc_addr(page), TB8);
                    } else if constexpr (T1B8) {
                        const uint32_t e0 = (r0 >> 4) * 32, m0 = EB + (r0 >> 4) * 2 * FB8;
                        const uint64_t dst = is_a ? oa.get_noc_addr(page) : ob.get_noc_addr(page);
                        noc_async_write(st + e0, dst + e0, 32);
                        noc_async_write(st + m0, dst + m0, 2 * FB8);
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
