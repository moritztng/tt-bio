// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1(i) writer -- drain the six per-block outputs (addr, fx, fy, fz, mask, sgn) so the host can
// grade every one of them against numpy independently. In the assembled kernel only the address CB
// leaves the core; the other five stay in L1 for the blend.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const auto d = TensorAccessor(dst_args, dst_addr, tile_bytes);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        for (uint32_t k = 0; k < 6; ++k) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(b * 6 + k, d, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
