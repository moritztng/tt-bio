// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// T1's work item u as (b, xt, h, yt): rows h * IL .. of x tile xt, y tile yt (trimul_tail.t1_unit mirrors it).
// PAIR (bfp8 at IL 8): items 2k and 2k + 1 are the two halves h = 2 hp, 2 hp + 1 of one face pair of the same
// (b, xt, yt), and the host gives every core whole pairs, so one core writes the face pair's exponent bytes once.
#pragma once

#include <cstdint>

namespace t1_unit {

struct Unit {
    uint32_t b, xt, h, yt;
};

template <uint32_t St, uint32_t H, bool PAIR>
inline Unit decode(uint32_t u) {
    const uint32_t b = u / (St * St * H), xt = (u / (St * H)) % St;
    if constexpr (PAIR) return {b, xt, 2 * ((u / (2 * St)) % (H / 2)) + (u & 1u), (u / 2) % St};
    return {b, xt, (u / St) % H, u % St};
}

}  // namespace t1_unit
