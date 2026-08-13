// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S1c -- is the per-transaction floor S1b measured a LATENCY that pipelines away, or an ISSUE cost
// that does not?
//
// S1 and S1b both called noc_async_read_barrier() once per assembled tile. That serialises: each
// iteration pays a full NoC round trip and no read can overlap with the next iteration's. A real
// reader issues many reads and barriers once, so the floor those screens measured may be an artifact
// of the harness rather than a property of the machine. If it pipelines, section 4.2(c) is not
// refuted and the design's per-row reader is fine.
//
// `barrier_every` assemblies share one barrier, and the destination cycles through that many slots so
// no in-flight read is overwritten by a later one. Sweeping it from 1 to 16 separates the two:
// latency amortises with depth, issue cost does not.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t total_bytes = get_compile_time_arg_val(1);   // bytes per assembly
    constexpr uint32_t nreads = get_compile_time_arg_val(2);
    constexpr uint32_t chunk = get_compile_time_arg_val(3);
    constexpr uint32_t barrier_every = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t npages = get_arg_val<uint32_t>(1);
    const uint32_t outer = get_arg_val<uint32_t>(2);
    const uint32_t core_base = get_arg_val<uint32_t>(3);

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    uint32_t bo[64];
    const uint32_t span = page_bytes - chunk;
    for (uint32_t r = 0; r < nreads; ++r) {
        bo[r] = get_arg_val<uint32_t>(4 + r) % (span + 1) & ~0x1Fu;
    }

    cb_reserve_back(cb_in, 1);
    const uint32_t w0 = get_write_ptr(cb_in);

    uint32_t base = core_base;
    uint32_t slot = 0;
    for (uint32_t i = 0; i < outer; ++i) {
        base += 7;
        if (base >= npages - 2) {
            base = 0;
        }
        uint32_t w = w0 + slot * total_bytes;
        for (uint32_t r = 0; r < nreads; ++r) {
            noc_async_read(s.get_noc_addr(base + (r & 1u), bo[r]), w, chunk);
            w += chunk;
        }
        if (++slot == barrier_every) {
            slot = 0;
            noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_in, 1);
}
