// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// E4 -- price the exact-trilinear gather, because the route was closed by a citation rather than a
// measurement.
//
// `relion-acc-backend` §4.6 killed exact-trilinear-on-device by quoting
// `ttnn-scatter-gather-per-element-limited` (~10-14 cycles per element, "1446x worse"), which was
// measured on a contiguous 45.1M-element `ttnn.gather`, a different op on a different access shape.
// Carried through this workload's own numbers the same citation gives 13x rather than a dead end, so
// it has to be measured on the real shape.
//
// The real shape. RELION's `CpuKernels::complex3D` interpolates each Fourier pixel from 8 corners of
// the padded model. Corners (x,y,z) and (x+1,y,z) are adjacent in memory -- a complex fp32 voxel is
// 8 B -- so one 16 B read fetches a corner PAIR and a whole pixel is 4 reads. The walk is not random:
// the projection steps `xp = e0*x + e1*y` along x, so consecutive pixels sit a fixed 3D increment
// apart. This kernel reproduces that: an accumulator per read slot advanced by a per-slot stride and
// wrapped into the volume, so addresses are correlated the way the real ones are and are neither
// contiguous nor uniformly random. A uniformly random arm would understate the route and a
// contiguous one would flatter it.
//
// The source is L1-resident. The padded model is 31.7 MB, which is 244 kB per core across 130 cores,
// so the route's premise is that it never touches DRAM. That premise is part of what is measured:
// pass an L1 tensor and the reads are core-to-core over the NoC.
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

    // One accumulator and one stride per read slot. The strides are odd multiples so no two slots
    // walk the same line, which is what 4 corner-pair reads of one pixel actually do.
    uint32_t acc[64], stride[64];
    const uint32_t span = npages * page_bytes - chunk;
    for (uint32_t r = 0; r < nreads; ++r) {
        acc[r] = (core_base * 4096u + get_arg_val<uint32_t>(4 + r)) % span;
        stride[r] = get_arg_val<uint32_t>(4 + nreads + r) | 1u;
    }

    cb_reserve_back(cb_in, 1);
    const uint32_t w0 = get_write_ptr(cb_in);

    uint32_t slot = 0;
    for (uint32_t i = 0; i < outer; ++i) {
        uint32_t w = w0 + slot * total_bytes;
        for (uint32_t r = 0; r < nreads; ++r) {
            // 16 B alignment: the NoC honours a read at 16 B from L1, and a corner pair is 16 B
            // aligned in the model by construction, so rounding here is the real address, not a
            // convenience.
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
    cb_push_back(cb_in, 1);
}
