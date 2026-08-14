// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S1d -- the last reader option. S1c showed the L1 per-row assembly pipelines to 1218 ns, but that
// L1 tensor was INTERLEAVED across 130 cores, so almost every read was a remote L1 read over the NoC
// and paid NoC latency. The design's strip is resident in the core's OWN L1, where no NoC transaction
// is needed at all: the RISC can address it directly.
//
// So: bulk-read one strip into a CB once, outside the timed loop, then assemble tiles out of it with
// plain RISC loads and stores at row-varying offsets. Arms:
//   0 LOCAL_VARY  32 x 64 B RISC copies at row-varying offsets -- the design's pattern, local.
//   1 LOCAL_BULK  one 2048 B RISC copy -- the same bytes, one contiguous move, so the arm above can
//                 be charged for its fragmentation rather than for the bytes.
//   2 NOOP        loop overhead only, so neither arm is credited with the loop.
// A local copy costs instruction issue, not NoC round trips, and 64 B is 16 uint32 words, so the
// question is whether ~512 load/store pairs beat a 1218 ns NoC assembly.
#include "api/dataflow/dataflow_api.h"

#define M_LOCAL_VARY 0
#define M_LOCAL_BULK 1
#define M_NOOP 2

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(1);
    constexpr uint32_t strip_pages = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t nrows = get_compile_time_arg_val(4);
    constexpr uint32_t chunk = get_compile_time_arg_val(5);      // bytes per row
    constexpr uint32_t mode = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t page0 = get_arg_val<uint32_t>(1);
    const uint32_t outer = get_arg_val<uint32_t>(2);

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // One bulk DRAM read of the whole strip, once, before the timed loop. This is the cost the
    // design amortises over every output tile the strip serves, and it is deliberately outside the
    // measurement -- what is being measured is the per-output-tile assembly out of it.
    cb_reserve_back(cb_src, strip_pages);
    const uint32_t sp = get_write_ptr(cb_src);
    uint32_t w = sp;
    for (uint32_t i = 0; i < strip_pages; ++i) {
        noc_async_read_page(page0 + i, s, w);
        w += page_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_src, strip_pages);

    const uint32_t strip_bytes = strip_pages * page_bytes;
    uint32_t bo[64];
    for (uint32_t r = 0; r < nrows; ++r) {
        bo[r] = get_arg_val<uint32_t>(3 + r) % (strip_bytes - chunk) & ~0x3u;
    }

    cb_reserve_back(cb_dst, 1);
    const uint32_t dp = get_write_ptr(cb_dst);
    const uint32_t words = chunk / 4;

    volatile uint32_t acc = 0;
    for (uint32_t i = 0; i < outer; ++i) {
        if constexpr (mode == M_NOOP) {
            acc += bo[i & (nrows - 1)];
        } else if constexpr (mode == M_LOCAL_BULK) {
            volatile uint32_t* d = (volatile uint32_t*)dp;
            const volatile uint32_t* q = (const volatile uint32_t*)(sp + bo[i & (nrows - 1)]);
            for (uint32_t k = 0; k < nrows * words; ++k) {
                d[k] = q[k];
            }
        } else {
            uint32_t off = 0;
            for (uint32_t r = 0; r < nrows; ++r) {
                volatile uint32_t* d = (volatile uint32_t*)(dp + off);
                const volatile uint32_t* q = (const volatile uint32_t*)(sp + bo[r]);
                for (uint32_t k = 0; k < words; ++k) {
                    d[k] = q[k];
                }
                off += chunk;
            }
        }
    }
    if constexpr (mode == M_NOOP) {
        *(volatile uint32_t*)dp = acc;
    }
    cb_push_back(cb_dst, 1);
    cb_pop_front(cb_src, strip_pages);
}
