// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// WS_PBO (= NT, n / 32 of a square pair tensor [B, n, n, C]): pair_swiglu_ws walks its row tiles in pair-bias order,
// the 32 rows i of one (b, ib, jb) output tile back to back, so an epilogue can assemble whole head-first bias tiles
// [B, H, n, n] on the core (tt_bio/pair_swiglu_ws.py `pbo`). The host splits super-units of 32 row tiles, so a
// core's [first, first + count) in this order holds whole super-units. Without WS_PBO the order is the identity.
#pragma once

inline uint32_t ws_row(uint32_t q) {
#ifdef WS_PBO
    constexpr uint32_t NT = WS_PBO;
    const uint32_t s = q >> 5;
    return ((s / NT) * 32 + (q & 31)) * NT + s % NT;   // s / NT = b * NT + ib, so (b * n + i) * NT + jb
#else
    return q;
#endif
}
