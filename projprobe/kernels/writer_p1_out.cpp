// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A drain writer parameterised by how many tiles a block emits, so each Phase 1 stage does not need
// its own. Reusing a writer whose per-block count did not match the compute kernel's output hung the
// blend arm: the writer waited on tiles that were never coming, which looks exactly like a kernel
// hang and is not one.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t per_block = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const auto d = TensorAccessor(dst_args, get_arg_val<uint32_t>(0), tile_bytes);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        for (uint32_t k = 0; k < per_block; ++k) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(b * per_block + k, d, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
