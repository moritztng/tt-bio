// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// T1's gather of one [a | b] channel tile: the IL tiles at the front of `src_cb` (compute's il order) moved face row
// by face row into the [B, C, S, S] tiles of a or b, NB output tiles per read barrier through a 2 * NB tile staging
// CB, then written to DRAM. writer.cpp runs it for every channel tile, or under T1SPLIT for the odd ones while
// reader.cpp runs it for the even ones from its own CBs: the tiles land in the same bytes either way.
// SKIP: the ABL bits 32 (gather) and 64 (writes) of writer.cpp (ABL itself is a macro there). B8: bfp8 a / b, at IL 32
// (whole tiles) or IL 8. A bfp8 tile keeps one exponent byte per face row, so an 8-row item owns 8 of a face's 16
// exponent bytes, which DRAM cannot take alone (16 B write alignment). At IL 8 the item order pairs the two items of a
// face pair (T1PAIR, unit.hpp), the first leaves its exponent bytes in a hold CB, and the second writes them for both:
// 32 B a tile at offset (r0 / 16) * 32. Mantissas go as 128 B rows like bf16's 256 B.
#pragma once

#include "api/dataflow/dataflow_api.h"

namespace t1_gather {

constexpr uint32_t TB = 2048, FB = 512, RB = 32;
// bfp8_b tile: 64 exponent bytes (face-major, one per face row), then 4 faces x 16 rows x 16 mantissa bytes.
constexpr uint32_t TB8 = 1088, EB = 64, FB8 = 256, RB8 = 16;
constexpr uint32_t HB = 32;   // held exponent bytes of one output tile: two faces, 16 rows

// 4 x 4 byte transpose: byte k of out[j] = byte j of w[k].
inline void transpose4(const uint32_t* w, uint32_t* out) {
    for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t sh = 8 * j;
        out[j] = ((w[0] >> sh) & 0xFFu) | (((w[1] >> sh) & 0xFFu) << 8) | (((w[2] >> sh) & 0xFFu) << 16) |
                 (((w[3] >> sh) & 0xFFu) << 24);
    }
}

template <uint32_t NB, uint32_t IL, uint32_t SKIP, bool B8>
struct Gather {
    static_assert(!B8 || IL == 32 || IL == 8, "bfp8 gathers whole tiles or paired 8-row items");
    static constexpr uint32_t SB = B8 ? TB8 : TB;   // staged tile stride
    uint32_t stage0, hold0;
    bool dirty[2];

    // `hold_cb`: IL 8 bfp8 only, HB bytes per output tile of every channel tile this core gathers.
    void init(uint32_t stage_cb, uint32_t hold_cb = 0) {
        cb_reserve_back(stage_cb, 2 * NB);
        stage0 = get_write_ptr(stage_cb);
        if constexpr (B8 && IL == 8) {
            cb_reserve_back(hold_cb, 1);
            hold0 = get_write_ptr(hold_cb);
        }
        dirty[0] = dirty[1] = false;
    }

    // Channel tile `page` (its first output tile, channel c = 0) of unit rows r0 .. r0 + IL - 1; `out` is a or b.
    // `slot`: this core's index of the channel tile, for the exponent hold (bfp8 IL 8).
    template <class Acc>
    void tile(uint32_t src_cb, const Acc& out, uint32_t page, uint32_t r0, uint32_t SS, uint32_t slot = 0) {
        // The staged tile keeps the unit's rows where the output has them, so one write (IL >= 16) or one per
        // face pair column (IL = 8) moves them.
        [[maybe_unused]] const uint32_t off0 = IL == 32 ? 0 : (r0 >> 4) * 2 * FB + (r0 & 15) * RB;
        [[maybe_unused]] const uint32_t wlen = IL == 32 ? TB : (IL == 16 ? 2 * FB : IL * RB);
        cb_wait_front(src_cb, IL);
        const uint32_t src = get_read_ptr(src_cb);
        [[maybe_unused]] const uint32_t page0 = page;
        [[maybe_unused]] const uint32_t hold = hold0 + slot * 32 * HB;
        [[maybe_unused]] const uint32_t hl = (r0 >> 3) & 1u;   // first or second item of the face pair
        if constexpr (B8 && IL == 8 && !(SKIP & 32)) {
            // Exponent byte of source tile il, row c, face side: ((c / 16) * 2 + side) * 16 + c % 16. Held at
            // hold[c][side * 16 + hl * 8 + il], the face row (r0 + il) % 16 of output tile c.
            for (uint32_t side = 0; side < 2; ++side)
                for (uint32_t c = 0; c < 32; c += 4) {
                    const uint32_t so = ((c >> 4) * 2 + side) * 16 + (c & 15);
                    for (uint32_t i0 = 0; i0 < 8; i0 += 4) {
                        uint32_t w[4], o[4];
                        for (uint32_t k = 0; k < 4; ++k)
                            w[k] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + (i0 + k) * TB8 + so);
                        transpose4(w, o);
                        for (uint32_t j = 0; j < 4; ++j)
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                hold + (c + j) * HB + side * 16 + hl * 8 + i0) = o[j];
                    }
                }
        }
        // Per tile: any other read on this core's NOC (reader.cpp's z pages) reprograms the read command buffer.
        noc_async_read_one_packet_set_state(get_noc_addr(stage0), B8 ? RB8 : RB);
        for (uint32_t c0 = 0; c0 < 32; c0 += NB) {
            const uint32_t grp = (c0 / NB) & 1u;
            const uint32_t gbase = stage0 + grp * NB * SB;
            if (dirty[grp]) noc_async_writes_flushed();
            for (uint32_t j = 0; j < NB && !(SKIP & 32); ++j) {
                const uint32_t c = c0 + j;
                const uint32_t st = gbase + j * SB;
                // source row c of tile il: face (c >> 4) * 2 (+1 for the right half), face row c & 15
                if constexpr (B8 && IL == 8) {
                    const uint32_t s0 = src + EB + (c >> 4) * 2 * FB8 + (c & 15) * RB8;
                    for (uint32_t il = 0; il < 8; ++il) {
                        const uint32_t r = r0 + il;
                        const uint32_t d0 = st + EB + (r >> 4) * 2 * FB8 + (r & 15) * RB8;
                        noc_async_read_one_packet_with_state(s0 + il * TB8, d0);
                        noc_async_read_one_packet_with_state(s0 + il * TB8 + FB8, d0 + FB8);
                    }
                } else if constexpr (B8) {
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
                const uint32_t st = gbase + j * SB;
                if constexpr (B8 && IL == 8) {
                    const uint32_t off = EB + (r0 >> 4) * 2 * FB8 + (r0 & 15) * RB8;
                    const uint64_t dst = out.get_noc_addr(page, off);
                    noc_async_write(st + off, dst, 8 * RB8);
                    noc_async_write(st + off + FB8, dst + FB8, 8 * RB8);
                } else if constexpr (B8) {
                    noc_async_write(st, out.get_noc_addr(page), TB8);
                } else {
                    const uint64_t dst = out.get_noc_addr(page, off0);
                    noc_async_write(st + off0, dst, wlen);
                    if constexpr (IL == 8) noc_async_write(st + off0 + FB, dst + FB, wlen);
                }
            }
            dirty[grp] = true;
        }
        if constexpr (B8 && IL == 8 && !(SKIP & 64)) {
            if (hl)   // the pair's second item: both items' exponents, 32 B a tile
                for (uint32_t c = 0; c < 32; ++c)
                    noc_async_write(hold + c * HB, out.get_noc_addr(page0 + c * SS, (r0 >> 4) * HB), HB);
        }
        noc_async_write_barrier();
        dirty[0] = dirty[1] = false;
        cb_pop_front(src_cb, IL);
    }
};

}  // namespace t1_gather
