// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// E8 -- fill one tile so the compute kernel has something real to multiply, then get out of the way.
// The screen measures the math unit, so the dataflow side does exactly one page read and stops.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t page = get_arg_val<uint32_t>(1);

    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);
    cb_reserve_back(cb_in, 1);
    noc_async_read_page(page, s, get_write_ptr(cb_in));
    noc_async_read_barrier();
    cb_push_back(cb_in, 1);
}
