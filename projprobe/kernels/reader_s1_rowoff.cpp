// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S1 -- the screen that can kill the design.
//
// Section 4.2(c) of the state doc puts ALL of the per-row variation of an affine warp into a
// per-row byte offset in the reader, and asserts that costs 32 address adds rather than a tile-op.
// If that is wrong -- if 32 row-varying reads cost materially more than 32 uniform ones -- then the
// per-row offset is a 32-way scalar loop in disguise and the design collapses to the rejected
// 4.6 ns/output diagonal-matmul variant. That is the whole point of measuring it before writing a
// compute kernel.
//
// Five arms, all assembling the same 2048 bytes (one bf16 32x32 tile) per iteration:
//   0 BULK       one 2048 B page read. The ideal, and the thing the ratio is against.
//   1 UNIF32     32 x 64 B plain reads, all at the same address. Isolates read-issue cost.
//   2 VARY32     32 x 64 B plain reads at row-varying (page, offset). The design's pattern.
//   3 STATE32    32 x 64 B via set_state once + with_state, row-varying low address. Same pattern,
//                the cheap issue path -- only usable when all 32 rows are in one DRAM bank, which
//                is a volume-layout constraint and is named as one.
//   4 STATE64    64 x 32 B via set_state/with_state. The honest variant: a tile's logical 32-wide
//                row is split across two 16x16 faces, so a contiguous source row lands in two
//                pieces.
// Offsets come from runtime args, so none of them can be constant-folded into an immediate.
#include "api/dataflow/dataflow_api.h"

#define M_BULK 0
#define M_UNIF32 1
#define M_VARY32 2
#define M_STATE32 3
#define M_STATE64 4

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t mode = get_compile_time_arg_val(2);
    constexpr uint32_t nrows = get_compile_time_arg_val(3);       // 32 or 64
    constexpr auto src_args = TensorAccessorArgs<4>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t npages = get_arg_val<uint32_t>(1);
    const uint32_t outer = get_arg_val<uint32_t>(2);
    const uint32_t chunk = tile_bytes / nrows;                    // 64 B (32 rows) or 32 B (64)

    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    // Per-row (page delta, byte offset), read out of runtime args into locals so the address
    // arithmetic is genuinely data-dependent.
    uint32_t pd[64], bo[64];
    for (uint32_t r = 0; r < nrows; ++r) {
        pd[r] = get_arg_val<uint32_t>(3 + 2 * r);
        bo[r] = get_arg_val<uint32_t>(4 + 2 * r);
    }

    cb_reserve_back(cb_in, 1);
    const uint32_t w0 = get_write_ptr(cb_in);

    uint32_t base = 0;
    for (uint32_t i = 0; i < outer; ++i) {
        base += 7;
        if (base >= npages - 2) {
            base = 0;
        }
        if constexpr (mode == M_BULK) {
            noc_async_read_page(base, s, w0);
        } else if constexpr (mode == M_UNIF32) {
            const uint64_t a = s.get_noc_addr(base, 0);
            uint32_t w = w0;
            for (uint32_t r = 0; r < nrows; ++r) {
                noc_async_read(a, w, chunk);
                w += chunk;
            }
        } else if constexpr (mode == M_VARY32) {
            uint32_t w = w0;
            for (uint32_t r = 0; r < nrows; ++r) {
                noc_async_read(s.get_noc_addr(base + pd[r], bo[r]), w, chunk);
                w += chunk;
            }
        } else {
            // STATE32 / STATE64: one set_state, then nrows issues varying only the low address.
            const uint64_t a0 = s.get_noc_addr(base, 0);
            noc_async_read_one_packet_set_state(a0, chunk);
            const uint32_t lo = (uint32_t)a0;
            uint32_t w = w0;
            for (uint32_t r = 0; r < nrows; ++r) {
                noc_async_read_one_packet_with_state(lo + bo[r], w);
                w += chunk;
            }
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_in, 1);
}
