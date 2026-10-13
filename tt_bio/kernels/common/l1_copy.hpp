// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Same-core L1 -> L1 copy on the RISC, for a data-movement kernel that also writes to DRAM.
//
// The rule it serves: a kernel never puts NoC reads and non-posted DRAM writes on the same NoC.
// The reblock-permute writers used to gather face rows with local NoC reads on the NoC that carried
// their DRAM tile writes, and on one Blackhole chip (qb2 chip 2, p300c, harvesting mask 0x12) that
// deadlocked inside a Protenix-v2 fold: every writer stuck issuing a gather read (watcher NATW),
// readers stuck on a DRAM read that never returned (NARW). The same pattern is tt-metal PR #59622.
// A copy whose source and destination are both in this core's L1 does not need the NoC at all.
// That chip later hung in other ops with this kernel disabled, so the chip is marginal and this
// rule alone does not make it safe. The copy stays where it costs nothing (reblock_permute,
// pair_transpose); the trimul gin writers keep their NoC gather, which is 8 % faster in --fast.
#pragma once

#include <cstdint>

namespace l1_copy {

// Copies BYTES (a multiple of 32) from src to dst, both 4-byte aligned local L1 addresses. Eight
// loads are issued before the eight stores so the L1 latency overlaps; the explicit temporaries also
// keep the compiler from turning the loop into a libc memcpy call.
template <uint32_t BYTES>
FORCE_INLINE void copy(uint32_t dst, uint32_t src) {
    static_assert(BYTES % 32 == 0, "l1_copy::copy moves whole 32-byte chunks");
    const tt_l1_ptr uint32_t* s = reinterpret_cast<const tt_l1_ptr uint32_t*>(src);
    tt_l1_ptr uint32_t* d = reinterpret_cast<tt_l1_ptr uint32_t*>(dst);
#pragma GCC unroll 4
    for (uint32_t k = 0; k < BYTES / 4; k += 8) {
        const uint32_t a0 = s[k + 0], a1 = s[k + 1], a2 = s[k + 2], a3 = s[k + 3];
        const uint32_t a4 = s[k + 4], a5 = s[k + 5], a6 = s[k + 6], a7 = s[k + 7];
        d[k + 0] = a0; d[k + 1] = a1; d[k + 2] = a2; d[k + 3] = a3;
        d[k + 4] = a4; d[k + 5] = a5; d[k + 6] = a6; d[k + 7] = a7;
    }
}

// Orders this RISC's stores before a NoC transaction that reads the same L1, and stops the compiler
// from sinking them past it.
FORCE_INLINE void publish() { asm volatile("fence" ::: "memory"); }

}  // namespace l1_copy
