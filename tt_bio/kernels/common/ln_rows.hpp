// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dataflow side of kernels/common/ln_prologue.hpp: a LayerNorm gamma / beta arrives as Kt tiles whose row 0
// carries the [K] vector. replicate_row0 copies that row into all 32 rows in place, once per program, so the
// prologue can apply gamma and beta as plain (unbroadcast) dest-reuse operands, the only dest-reuse form
// Wormhole's unpacker supports. A ROW broadcast of the replicated tile reads the same row 0 as before.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// n tiles of tile_bytes each (bf16 or fp32, four 16x16 faces) from L1 address addr. Call after the read
// barrier that landed them, before the CB push.
FORCE_INLINE void replicate_row0(uint32_t addr, uint32_t n, uint32_t tile_bytes) {
    const uint32_t face = tile_bytes / 4, words = face / 16 / 4;          // bytes per face, words per face row
    for (uint32_t t = 0; t < n; ++t, addr += tile_bytes) {
        for (uint32_t f = 0; f < 4; ++f) {
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr + (f & 1) * face);
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr + f * face);
            for (uint32_t r = f < 2 ? 1 : 0; r < 16; ++r)
                for (uint32_t i = 0; i < words; ++i) dst[r * words + i] = src[i];
        }
    }
}
