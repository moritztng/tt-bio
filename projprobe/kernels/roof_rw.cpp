// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// B0 -- the DRAM read, write and read-modify-write roofs, measured SEPARATELY.
//
// Every roof quoted so far in this program came from `ttnn.add` on a large tensor, which is two
// reads and one write. That is a mixed roof and it cannot answer the backprojection question,
// because backprojection's irreducible traffic is on the WRITE side (the accumulated volume) and
// the forward's was on the read side. A mixed roof borrowed across that boundary would be wrong in
// an unknown direction.
//
// Each core owns a disjoint contiguous page range so no two cores contend for a page, and `be`
// pages are in flight before a barrier so a per-transaction latency amortises exactly as it does in
// the built kernels. Mode 2 issues the reads for the whole group, barriers, then issues the writes:
// that is the cheapest honest RMW, since the data must arrive before it can go back.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t xact_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t mode = get_compile_time_arg_val(3);  // 0 read, 1 write, 2 read-modify-write
    constexpr uint32_t be = get_compile_time_arg_val(4);
    constexpr auto acc_args = TensorAccessorArgs<5>();

    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t page0 = get_arg_val<uint32_t>(1);   // first page this core owns
    const uint32_t npage = get_arg_val<uint32_t>(2);   // pages this core owns
    const uint32_t outer = get_arg_val<uint32_t>(3);   // multiple of be
    const uint32_t pattern = get_arg_val<uint32_t>(4);

    const auto acc = TensorAccessor(acc_args, addr, page_bytes);

    cb_reserve_back(cb, 1);
    const uint32_t l1 = get_write_ptr(cb);

    // Fill the staging slots so a write arm writes something checkable rather than whatever the CB
    // happened to hold. An arm whose bytes could be anything is not a measurement of writing them.
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < (be * page_bytes) / 4; ++i) {
        p[i] = pattern;
    }

    constexpr uint32_t nsub = page_bytes / xact_bytes;
    uint32_t page = page0;
    for (uint32_t i = 0; i < outer; i += be) {
        for (uint32_t k = 0; k < be; ++k) {
            const uint32_t lp = l1 + k * page_bytes;
            for (uint32_t s = 0; s < nsub; ++s) {
                const uint64_t na = acc.get_noc_addr(page + k, s * xact_bytes);
                if constexpr (mode == 1) {
                    noc_async_write(lp + s * xact_bytes, na, xact_bytes);
                } else {
                    noc_async_read(na, lp + s * xact_bytes, xact_bytes);
                }
            }
        }
        if constexpr (mode == 1) {
            noc_async_write_barrier();
        } else {
            noc_async_read_barrier();
        }
        if constexpr (mode == 2) {
            for (uint32_t k = 0; k < be; ++k) {
                const uint32_t lp = l1 + k * page_bytes;
                for (uint32_t s = 0; s < nsub; ++s) {
                    noc_async_write(lp + s * xact_bytes,
                                    acc.get_noc_addr(page + k, s * xact_bytes), xact_bytes);
                }
            }
            noc_async_write_barrier();
        }
        page += be;
        if (page + be > page0 + npage) {
            page = page0;
        }
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    cb_push_back(cb, 1);
}
