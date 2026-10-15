// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// T1's bf16 face-row gather for one channel tile q, channels C0 .. C1 - 1 of it: compute's IL tiles at
// `src` hold [channel c, y] of rows r0 .. r0 + IL - 1; output tile c (page `page` + (c - C0) * SS of `a`
// or `b`) takes row c of each in its rows r0 + il. NB tiles at a time are assembled in one half of the
// 2 * NB tile stage at `stage0` and written while the other half gathers. Both data-movement kernels run
// it (SPLIT): the writer on channels 0..15, the reader on 16..31, each on its own NOC.
#pragma once
#include "api/dataflow/dataflow_api.h"

template <uint32_t IL, uint32_t NB, uint32_t C0, uint32_t C1, uint32_t SS, uint32_t ABLM, class A, class B>
inline void t1_gather_q(uint32_t src, uint32_t stage0, uint32_t page, bool is_a, const A& oa, const B& ob,
                        uint32_t r0, bool (&dirty)[2], volatile tt_l1_ptr uint32_t* done = nullptr,
                        uint32_t done_val = 0) {
    constexpr uint32_t TB = 2048, FB = 512, RB = 32;
    // The unit's rows r0 .. r0 + IL - 1 of each output tile: the staged tile keeps them where the output
    // has them, so one write (IL >= 16) or one per face pair column (IL = 8) moves them.
    const uint32_t off0 = IL == 32 ? 0 : (r0 >> 4) * 2 * FB + (r0 & 15) * RB;
    constexpr uint32_t wlen = IL == 32 ? TB : (IL == 16 ? 2 * FB : IL * RB);
    noc_async_read_one_packet_set_state(get_noc_addr(stage0), RB);
    for (uint32_t c0 = C0; c0 < C1; c0 += NB) {
        const uint32_t grp = ((c0 - C0) / NB) & 1u;
        const uint32_t gbase = stage0 + grp * NB * TB;
        if (dirty[grp]) noc_async_writes_flushed();
        for (uint32_t j = 0; j < NB && !(ABLM & 32); ++j) {
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
        noc_async_read_barrier();
        // The last read out of `src` has landed: whoever pops it may.
        if (done != nullptr && c0 + NB >= C1) noc_semaphore_set(done, done_val);
        for (uint32_t j = 0; j < NB; ++j, page += SS) {
            if constexpr (ABLM & 64) continue;
            const uint32_t st = gbase + j * TB;
            const uint64_t dst = is_a ? oa.get_noc_addr(page, off0) : ob.get_noc_addr(page, off0);
            noc_async_write(st + off0, dst, wlen);
            if constexpr (IL == 8) noc_async_write(st + off0 + FB, dst + FB, wlen);
        }
        dirty[grp] = true;
    }
}
