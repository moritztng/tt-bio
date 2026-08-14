// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S1b reader. Loads NPAGES fp32 tiles into c_0 exactly once, before the compute kernel's timed
// loop starts. Nothing is read inside the loop: the whole point of the screen is to measure
// arithmetic against L1 traffic with DRAM removed, so DRAM appears once per program launch and is
// amortised over `outer` iterations.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t npages = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t page0 = get_arg_val<uint32_t>(1);

    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    cb_reserve_back(cb_in, npages);
    uint32_t w = get_write_ptr(cb_in);
    for (uint32_t i = 0; i < npages; ++i) {
        noc_async_read_page(page0 + i, s, w);
        w += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_in, npages);
}
