// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dump the output CB once, from one core only, so the arm can be checked for having actually
// packed. Not part of the timed slope: it runs after the loop has finished.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t do_write = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t OUT_SLOTS = get_compile_time_arg_val(1);

    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto d = TensorAccessor(dst_args, dst_addr);

    cb_wait_front(cb_id, OUT_SLOTS);
    if (do_write) {
        uint32_t rd = get_read_ptr(cb_id);
        const uint32_t tile_bytes = get_tile_size(cb_id);
        for (uint32_t i = 0; i < OUT_SLOTS; ++i) {
            noc_async_write_page(i, d, rd);
            rd += tile_bytes;
        }
        noc_async_write_barrier();
    }
    cb_pop_front(cb_id, OUT_SLOTS);
}
