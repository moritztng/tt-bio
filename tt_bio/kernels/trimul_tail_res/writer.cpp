// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail, weights resident: writes this core's output rows, one block of Mb x Nt tiles.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t Mb = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    const uint32_t tb = get_tile_size(out_cb);
    for (uint32_t b = 0; b < nblocks; ++b) {
        const uint32_t m = m0 + b * Mb;
        cb_wait_front(out_cb, Mb * Nt);
        uint32_t r = get_read_ptr(out_cb);
        for (uint32_t t = 0; t < Mb * Nt; ++t, r += tb) noc_async_write_page(m * Nt + t, out, r);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, Mb * Nt);
    }
    noc_async_write_barrier();
}
