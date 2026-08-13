// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S1b -- the diagnostic S1 forced. S1 found the address arithmetic free and the read fragmentation
// ruinous: one 2048 B read costs 649 ns, thirty-two 64 B reads cost 8627 ns. That is a 15.5x penalty
// for fragmentation, and it does not tell us WHICH of the two candidate mechanisms is responsible:
//   * per-read ISSUE cost on the dataflow RISC, in which case the fix is fewer, larger reads;
//   * DRAM/NoC efficiency at small transfer granularity, in which case an L1-resident source is
//     immune and the fix is to make the source L1-resident and keep the per-row offsets.
// Those two have opposite design consequences, so the screen separates them: the same total bytes
// split every way from 1 read to 128, against a DRAM source and against an L1 source. Issue cost is
// flat in source memory; granularity cost is not.
//
// Every core reads from a different base page, so no arm can be inflated by 130 cores hitting one
// hot DRAM page -- an artifact S1 did not control for.
#include "api/dataflow/dataflow_api.h"

#define M_PLAIN 0
#define M_STATE 1

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t total_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t nreads = get_compile_time_arg_val(2);
    constexpr uint32_t chunk = get_compile_time_arg_val(3);      // total_bytes / nreads
    constexpr uint32_t mode = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t npages = get_arg_val<uint32_t>(1);
    const uint32_t outer = get_arg_val<uint32_t>(2);
    const uint32_t core_base = get_arg_val<uint32_t>(3);

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // Row-varying byte offsets, from runtime args so nothing folds to an immediate. A shear's
    // offsets are monotone in the row; these are, and they stay inside one page.
    uint32_t bo[128];
    const uint32_t span = page_bytes - chunk;
    for (uint32_t r = 0; r < nreads; ++r) {
        bo[r] = get_arg_val<uint32_t>(4 + r) % (span + 1) & ~0x1Fu;
    }

    cb_reserve_back(cb_in, 1);
    const uint32_t w0 = get_write_ptr(cb_in);

    uint32_t base = core_base;
    for (uint32_t i = 0; i < outer; ++i) {
        base += 7;
        if (base >= npages - 2) {
            base = 0;
        }
        if constexpr (mode == M_PLAIN) {
            uint32_t w = w0;
            for (uint32_t r = 0; r < nreads; ++r) {
                noc_async_read(s.get_noc_addr(base, bo[r]), w, chunk);
                w += chunk;
            }
        } else {
            const uint64_t a0 = s.get_noc_addr(base, 0);
            noc_async_read_one_packet_set_state(a0, chunk);
            const uint32_t lo = (uint32_t)a0;
            uint32_t w = w0;
            for (uint32_t r = 0; r < nreads; ++r) {
                noc_async_read_one_packet_with_state(lo + bo[r], w);
                w += chunk;
            }
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_in, 1);
}
