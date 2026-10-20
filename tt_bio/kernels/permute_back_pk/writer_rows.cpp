// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tile_rows: unit u = (i, jb)'s 32 output tiles are out[i, jb*32 .. jb*32 + 31, p*32 .. p*32 + 31], p = 0..31:
// 32 consecutive pages from u*32, WB per flush.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WB = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    const uint32_t tb = get_tile_size(out_cb);

    for (uint32_t page = u0 * 32; page < (u0 + nunits) * 32;) {
        cb_wait_front(out_cb, WB);
        uint32_t src = get_read_ptr(out_cb);
        for (uint32_t k = 0; k < WB; ++k, ++page, src += tb) noc_async_write_page(page, o, src);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, WB);
    }
    noc_async_write_barrier();
}
