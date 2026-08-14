// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// FFT writer. Drains the output CB in chunks as pass 2 produces it.
//
// CHUNK is the lever, and it is a barrier-count lever rather than a bandwidth one. The first
// version wrote two tiles and then called noc_async_write_barrier, which is 64 full round trips to
// DRAM completion per image, each one stalling the writer while the compute kernel waits on CB
// space behind it. Raising the chunk lets that many writes be in flight at once and cuts the
// barrier count by CHUNK/2. The barrier still has to happen before cb_pop_front, because popping
// releases L1 the NoC is reading out of.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t ntile = get_compile_time_arg_val(2);
    constexpr uint32_t nimg = get_compile_time_arg_val(3);
    constexpr uint32_t chunk = get_compile_time_arg_val(4);
    constexpr auto o_args = TensorAccessorArgs<5>();

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t page0 = get_arg_val<uint32_t>(1);

    const auto oa = TensorAccessor(o_args, o_addr, tile_bytes);

    uint32_t page = page0;
    for (uint32_t img = 0; img < nimg; ++img) {
        for (uint32_t i = 0; i < ntile; i += chunk) {
            cb_wait_front(cb_o, chunk);
            uint32_t r = get_read_ptr(cb_o);
            for (uint32_t k = 0; k < chunk; ++k) {
                noc_async_write_page(page + i + k, oa, r);
                r += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_o, chunk);
        }
        page += ntile;
    }
}
