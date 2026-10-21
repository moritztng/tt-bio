// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tile_rows: x [R*32, L*32] -> out [R, L, 1024]. Per unit u = (i, jb): the 32 tiles x[i, jb*32 + r], r = 0..31,
// which are 32 consecutive pages, in groups of 8.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    const uint32_t tb = get_tile_size(in_cb);

    for (uint32_t page = u0 * 32; page < (u0 + nunits) * 32;) {
        cb_reserve_back(in_cb, 8);
        uint32_t dst = get_write_ptr(in_cb);
        for (uint32_t k = 0; k < 8; ++k, ++page, dst += tb) noc_async_read_page(page, x, dst);
        noc_async_read_barrier();
        cb_push_back(in_cb, 8);
    }
}
