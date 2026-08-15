// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1(ii) writer -- drain the eight gathered slot tiles per block so the host can grade each
// corner independently. In the assembled kernel these never leave L1; the blend consumes them.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_slot = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const auto d = TensorAccessor(dst_args, dst_addr, tile_bytes);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_wait_front(cb_slot, 8);
        const uint32_t p = get_read_ptr(cb_slot);
        for (uint32_t s = 0; s < 8; ++s) {
            noc_async_write_page(b * 8 + s, d, p + s * tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_slot, 8);
    }
}
