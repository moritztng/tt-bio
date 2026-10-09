// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pair layer norm writer: Wt tiles per tile-row, in the output CB's format.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_row = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 16;
    constexpr auto d_args = TensorAccessorArgs<1>();
    const auto sd = TensorAccessor(d_args, dst_addr);
    const uint32_t tb = get_tile_size(cb_out);
    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t page = (first_row + r) * Wt;
        cb_wait_front(cb_out, Wt);
        uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < Wt; ++j) { noc_async_write(rd, sd.get_noc_addr(page + j), tb); rd += tb; }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, Wt);
    }
    noc_async_write_barrier();
}
