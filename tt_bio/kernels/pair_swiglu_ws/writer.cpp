// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_swiglu_ws writer. Drains c_16, CT tiles per row tile, to pages [r*CT, (r+1)*CT) of the output.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 16;
    constexpr auto o_args = TensorAccessorArgs<1>();
    const auto so = TensorAccessor(o_args, out_addr);
    const uint32_t to = get_tile_size(cb_out);
    for (uint32_t r = first; r < first + count; ++r) {
        cb_wait_front(cb_out, CT);
        uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < CT; ++c, rd += to) {
            noc_async_write_page(r * CT + c, so, rd);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, CT);
    }
}
