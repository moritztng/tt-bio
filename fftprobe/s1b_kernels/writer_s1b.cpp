// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S1b writer. One tile per core, written once after the timed loop. Its only job is to make the
// result observable so the arms can be checked for correctness rather than only timed -- an arm
// whose arithmetic the compiler could have elided is not a measurement of that arithmetic.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t page = get_arg_val<uint32_t>(1);

    const auto d = TensorAccessor(dst_args, dst_addr, tile_bytes);

    cb_wait_front(cb_out, 1);
    noc_async_write_page(page, d, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
