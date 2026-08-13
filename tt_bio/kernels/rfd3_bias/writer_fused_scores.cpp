// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RFD3 fused attention-score writer: drains the compute kernel's fp32 tiles to DRAM.
//
// It walks the same (head, tile-row) bands in the same order as the reader, so the n-th tile out
// of the compute kernel is the n-th page this kernel writes and no page number has to be carried
// through a CB.
//
// The window is WINDOW tiles deep: a page cannot be popped until its write has landed (the
// compute kernel would pack into it), so a barrier per tile would serialise the write latency
// 339 times per core at the production shape. Waiting for WINDOW, issuing WINDOW writes, then one
// barrier, amortises it. Page addresses are computed from the ring base and this kernel's own
// counter rather than from get_read_ptr, because Jt is not a multiple of WINDOW and a window can
// straddle the ring wrap once the tail has pushed the front pointer out of alignment.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known device-wedge
// cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t start_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t It = get_compile_time_arg_val(1);
    constexpr uint32_t Jt = get_compile_time_arg_val(2);
    constexpr uint32_t WINDOW = get_compile_time_arg_val(3);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(4);  // cb_out depth, power of two
    constexpr auto out_args = TensorAccessorArgs<5>();

    constexpr uint32_t OUT_TILE_BYTES = 32 * 32 * 4;

    const auto s_out = TensorAccessor(out_args, out_addr);
    const uint32_t out_base = get_read_ptr(cb_out);

    uint32_t popped = 0;
    const uint32_t end_group = start_group + num_groups;
    for (uint32_t group = start_group; group < end_group; ++group) {
        const uint32_t h = group / It;
        const uint32_t it = group - h * It;
        uint32_t page = (h * It + it) * Jt;

        uint32_t jt = 0;
        while (jt < Jt) {
            uint32_t n = Jt - jt;
            if (n > WINDOW) {
                n = WINDOW;
            }
            cb_wait_front(cb_out, n);
            for (uint32_t w = 0; w < n; ++w) {
                noc_async_write(out_base + ((popped + w) & (SLOTS - 1)) * OUT_TILE_BYTES,
                                s_out.get_noc_addr(page + w), OUT_TILE_BYTES);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, n);
            popped += n;
            page += n;
            jt += n;
        }
    }
}
