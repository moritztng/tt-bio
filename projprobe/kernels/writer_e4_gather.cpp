// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// E4b -- the gather is issue-bound at ~47 cycles per read on ONE dataflow RISC. A Tensix has two that
// can issue NoC reads, so the question this kernel answers is whether the second one is free.
//
// Same loop as reader_e4_gather.cpp, run in the writer slot (BRISC) against its own scratch CB and
// its own starting phase, so the two RISCs walk different lines and neither reads the other's
// destination. If the per-core rate doubles, the issue cost is per-RISC and the exact-trilinear route
// clears its bar with room; if it does not, the bottleneck is the core's NoC port and one RISC is all
// there is.
//
// The single output write still happens, after the loop, so the arm remains observable.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t total_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t nreads = get_compile_time_arg_val(2);
    constexpr uint32_t chunk = get_compile_time_arg_val(3);
    constexpr uint32_t barrier_every = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t push_early = get_compile_time_arg_val(8);
    constexpr auto src_args = TensorAccessorArgs<9>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t npages = get_arg_val<uint32_t>(1);
    const uint32_t outer = get_arg_val<uint32_t>(2);
    const uint32_t core_base = get_arg_val<uint32_t>(3);
    const uint32_t dst_addr = get_arg_val<uint32_t>(4);
    const uint32_t page = get_arg_val<uint32_t>(5);

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    uint32_t acc[64], stride[64];
    const uint32_t span = npages * page_bytes - chunk;
    for (uint32_t r = 0; r < nreads; ++r) {
        // A different phase from the reader's, so the two RISCs are not chasing the same lines.
        acc[r] = (core_base * 4096u + get_arg_val<uint32_t>(6 + r) + span / 2u) % span;
        stride[r] = get_arg_val<uint32_t>(6 + nreads + r) | 1u;
    }

    cb_reserve_back(cb_scratch, 1);
    const uint32_t w0 = get_write_ptr(cb_scratch);
    if (push_early) {
        cb_push_back(cb_scratch, 1);
    }

    uint32_t slot = 0;
    for (uint32_t i = 0; i < outer; ++i) {
        uint32_t w = w0 + slot * total_bytes;
        for (uint32_t r = 0; r < nreads; ++r) {
            const uint32_t a = acc[r] & ~0xFu;
            noc_async_read(s.get_noc_addr(a / page_bytes, a % page_bytes), w, chunk);
            w += chunk;
            acc[r] += stride[r];
            if (acc[r] >= span) {
                acc[r] -= span;
            }
        }
        if (++slot == barrier_every) {
            slot = 0;
            noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();

    const auto d = TensorAccessor(dst_args, dst_addr, tile_bytes);
    cb_wait_front(cb_out, 1);
    noc_async_write_page(page, d, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
