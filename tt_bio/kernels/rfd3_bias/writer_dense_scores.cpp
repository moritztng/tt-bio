// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RFD3 dense attention-score writer: drains the compute kernel's fp32 tiles to DRAM.
//
// The dense path splits work as a flat page range rather than as (head, tile-row) bands, so this
// is writer_fused_scores.cpp's loop with the band arithmetic removed: the n-th tile out of the
// compute kernel is page start_page + n. Kept separate from that kernel rather than folded into it
// with It = 1, because "a band of Jt pages" and "a range of num_pages pages" want different
// coalescing and the sparse path's own writer is on a bit-exact gate.
//
// WINDOW pages per barrier: a page cannot be popped until its write has landed, so one barrier per
// page would serialise the write latency 60 times per core at the production shape. The ring slot
// is tracked here rather than read from the CB, because a window can straddle the wrap.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known device-wedge
// cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(1);   // cb_out depth, power of two
    constexpr uint32_t WINDOW = get_compile_time_arg_val(2);  // pages per barrier, <= SLOTS
    constexpr auto out_args = TensorAccessorArgs<3>();

    constexpr uint32_t OUT_TILE_BYTES = 32 * 32 * 4;

    const auto s_out = TensorAccessor(out_args, out_addr);
    const uint32_t out_base = get_read_ptr(cb_out);

    uint32_t popped = 0;
    uint32_t page = start_page;
    uint32_t left = num_pages;
    while (left) {
        uint32_t n = (left < WINDOW) ? left : WINDOW;
        cb_wait_front(cb_out, n);
        for (uint32_t w = 0; w < n; ++w) {
            noc_async_write(out_base + ((popped + w) & (SLOTS - 1)) * OUT_TILE_BYTES,
                            s_out.get_noc_addr(page + w), OUT_TILE_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, n);
        popped += n;
        page += n;
        left -= n;
    }
}
