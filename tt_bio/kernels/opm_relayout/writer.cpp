// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// opm_relayout writer: streams each group's 32 finished tiles to output pages 32g .. 32g + 31.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t TILES = 32;
    constexpr uint32_t TILE_BYTES = 2048;

    const auto s = TensorAccessor(dst_args, dst_addr);
    for (uint32_t g = first_group; g < first_group + num_groups; ++g) {
        cb_wait_front(out_cb, TILES);
        uint32_t l1 = get_read_ptr(out_cb);
        for (uint32_t c = 0; c < TILES; ++c) {
            noc_async_write(l1, s.get_noc_addr(g * TILES + c), TILE_BYTES);
            l1 += TILE_BYTES;
        }
        noc_async_write_barrier();
        cb_pop_front(out_cb, TILES);
    }
}
