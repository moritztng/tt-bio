// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fourier-slice projection, stage 2 reader: assemble one row-major source window per output tile,
// each row starting at its own byte offset, and load the pass's fixed operands once.
//
// Section 4.2(c) as the screens left it. S1: the per-row address arithmetic is free (row-varying reads
// cost 0.855x uniform ones, and the stateful issue path buys nothing). S1c: the per-transaction cost
// is a LATENCY that pipelines on an L1-resident source (1.44x from barriering every 4th assembly,
// 38.1 ns/read) and does NOT pipeline from DRAM (1.01x, 277 ns/read). S1e: a 128 B row read costs
// 1.118x a 64 B one, so the 64-wide window the +1/+2 interpolation shifts need is nearly free, while
// splitting it into two 32-wide assemblies would cost 2.65x.
//
// THE ALIGNMENT CONSTRAINT IS THE ONE THAT SHAPES THIS KERNEL. A per-row offset is only honoured at
// 64 B granularity from a DRAM source and 16 B from an L1 source, and giving the destination the
// source's misalignment does not help -- the constraint is absolute on both addresses, not relative
// (all three measured in projprobe/fslice_align.py). With bf16 that is 32 elements from DRAM and 8
// from L1, so the source must be L1-resident for the offsets to be useful at all, and the residual
// 0..7 element shift is not expressible here.
//
// The destination is CONTIGUOUS row-major in L1 -- 32 rows x 64 elements -- which is what
// tilize_block consumes. Reading row-major and tilizing on the compute engine is what keeps each row
// a single transaction; a TILE_LAYOUT source would split every logical row across two 16x16 faces.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t win_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t nrows = get_compile_time_arg_val(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);      // source row pitch
    constexpr uint32_t ntiles = get_compile_time_arg_val(4);         // CB tiles per assembly
    constexpr uint32_t barrier_every = get_compile_time_arg_val(5);  // S1c optimum is 4
    constexpr uint32_t mode = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(7);
    constexpr uint32_t cb_frac = get_compile_time_arg_val(8);
    constexpr uint32_t nsel = get_compile_time_arg_val(9);           // 3 * ntiles
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t nfrac = get_compile_time_arg_val(11);   // 2 for modes 1/4, 3 for mode 5
    constexpr auto src_args = TensorAccessorArgs<12>();
    constexpr auto sel_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto frac_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);
    const uint32_t row0 = get_arg_val<uint32_t>(2);

    uint32_t bo[32];
    for (uint32_t r = 0; r < nrows; ++r) {
        bo[r] = get_arg_val<uint32_t>(3 + r);
    }
    const uint32_t sel_addr = get_arg_val<uint32_t>(3 + nrows);
    const uint32_t frac_addr = get_arg_val<uint32_t>(4 + nrows);

    const auto s = TensorAccessor(src_args, src_addr, row_bytes);

    // The selection matrices and the two fraction vectors are fixed for the whole orientation, so they
    // are loaded once here rather than per output tile. Without this the compute kernel's
    // cb_wait_front on them never returns and the program deadlocks.
    if constexpr (mode != 0) {
        const auto sa = TensorAccessor(sel_args, sel_addr, tile_bytes);
        const auto fa = TensorAccessor(frac_args, frac_addr, tile_bytes);
        cb_reserve_back(cb_sel, nsel);
        uint32_t w = get_write_ptr(cb_sel);
        for (uint32_t i = 0; i < nsel; ++i) {
            noc_async_read_page(i, sa, w);
            w += tile_bytes;
        }
        cb_reserve_back(cb_frac, nfrac);
        uint32_t wf = get_write_ptr(cb_frac);
        for (uint32_t i = 0; i < nfrac; ++i) {
            noc_async_read_page(i, fa, wf);
            wf += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_sel, nsel);
        cb_push_back(cb_frac, nfrac);
    }

    // Assemblies are grouped so the barrier lands BEFORE the push. Pushing on an unfinished read would
    // let the compute engine consume bytes that have not arrived; the grouping is also what buys
    // S1c's 1.44x, one barrier per `barrier_every` assemblies rather than one per assembly.
    uint32_t b = 0;
    while (b < nblocks) {
        uint32_t g = nblocks - b;
        if (g > barrier_every) {
            g = barrier_every;
        }
        cb_reserve_back(cb_src, g * ntiles);
        uint32_t w = get_write_ptr(cb_src);
        for (uint32_t i = 0; i < g; ++i) {
            for (uint32_t r = 0; r < nrows; ++r) {
                noc_async_read(s.get_noc_addr(row0 + r, bo[r]), w, win_bytes);
                w += win_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_src, g * ntiles);
        b += g;
    }
}
