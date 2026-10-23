// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// T1's gather of one [a | b] channel tile: the IL tiles at the front of `src_cb` (compute's il order) moved face row
// by face row into the [B, C, S, S] tiles of a or b, NB output tiles per read barrier through a 2 * NB tile staging
// CB, then written to DRAM. writer.cpp runs it for every channel tile, or under T1SPLIT for the odd ones while
// reader.cpp runs it for the even ones from its own CBs: the tiles land in the same bytes either way.
// SKIP: the ABL bits 32 (gather) and 64 (writes) of writer.cpp (ABL itself is a macro there). T1B8: bfp8 a / b, whole tiles (IL 32).
#pragma once

#include "api/dataflow/dataflow_api.h"

namespace t1_gather {

constexpr uint32_t TB = 2048, FB = 512, RB = 32;
// bfp8_b tile: 64 exponent bytes (face-major, one per face row), then 4 faces x 16 rows x 16 mantissa bytes.
constexpr uint32_t TB8 = 1088, EB = 64, FB8 = 256, RB8 = 16;

template <uint32_t NB, uint32_t IL, uint32_t SKIP, bool B8>
struct Gather {
    uint32_t stage0;
    bool dirty[2];

    void init(uint32_t stage_cb) {
        cb_reserve_back(stage_cb, 2 * NB);
        stage0 = get_write_ptr(stage_cb);
        dirty[0] = dirty[1] = false;
    }

    // Channel tile `page` (its first output tile, channel c = 0) of unit rows r0 .. r0 + IL - 1; `out` is a or b.
    template <class Acc>
    void tile(uint32_t src_cb, const Acc& out, uint32_t page, uint32_t r0, uint32_t SS) {
        // The staged tile keeps the unit's rows where the output has them, so one write (IL >= 16) or one per
        // face pair column (IL = 8 or 4) moves them.
        [[maybe_unused]] const uint32_t off0 = IL == 32 ? 0 : (r0 >> 4) * 2 * FB + (r0 & 15) * RB;
        [[maybe_unused]] const uint32_t wlen = IL == 32 ? TB : (IL == 16 ? 2 * FB : IL * RB);
        cb_wait_front(src_cb, IL);
        const uint32_t src = get_read_ptr(src_cb);
        // Per tile: any other read on this core's NOC (reader.cpp's z pages) reprograms the read command buffer.
        noc_async_read_one_packet_set_state(get_noc_addr(stage0), B8 ? RB8 : RB);
        for (uint32_t c0 = 0; c0 < 32; c0 += NB) {
            const uint32_t grp = (c0 / NB) & 1u;
            const uint32_t gbase = stage0 + grp * NB * TB;
            if (dirty[grp]) noc_async_writes_flushed();
            for (uint32_t j = 0; j < NB && !(SKIP & 32); ++j) {
                const uint32_t c = c0 + j;
                const uint32_t st = gbase + j * TB;
                // source row c of tile il: face (c >> 4) * 2 (+1 for the right half), face row c & 15
                if constexpr (B8) {
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
                } else {
                    // destination row r0 + il, the same face arithmetic
                    uint32_t s0 = src + (c >> 4) * 2 * FB + (c & 15) * RB;
                    for (uint32_t il = 0; il < IL; ++il, s0 += TB) {
                        const uint32_t r = r0 + il;
                        const uint32_t d0 = st + (r >> 4) * 2 * FB + (r & 15) * RB;
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s0 + FB, d0 + FB);
                    }
                }
            }
            noc_async_read_barrier();
            for (uint32_t j = 0; j < NB; ++j, page += SS) {
                if constexpr (SKIP & 64) continue;
                const uint32_t st = gbase + j * TB;
                if constexpr (B8) {
                    noc_async_write(st, out.get_noc_addr(page), TB8);
                } else {
                    const uint64_t dst = out.get_noc_addr(page, off0);
                    noc_async_write(st + off0, dst, wlen);
                    if constexpr (IL <= 8) noc_async_write(st + off0 + FB, dst + FB, wlen);
                }
            }
            dirty[grp] = true;
        }
        noc_async_write_barrier();
        dirty[0] = dirty[1] = false;
        cb_pop_front(src_cb, IL);
    }
};

}  // namespace t1_gather
