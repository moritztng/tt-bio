// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fill the input CB once with NT tiles, then exit. Every core reads the same NT pages: the DRAM
// traffic is one-off and outside the rate slope, and having all cores hold identical data keeps the
// multi-core arm comparable to the single-core one.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id2 = get_compile_time_arg_val(2);
    constexpr uint32_t fill2 = get_compile_time_arg_val(3);

    constexpr auto src_args = TensorAccessorArgs<4>();
    const auto s = TensorAccessor(src_args, src_addr);

    cb_reserve_back(cb_id, NT);
    uint32_t wr = get_write_ptr(cb_id);
    const uint32_t tile_bytes = get_tile_size(cb_id);
    for (uint32_t i = 0; i < NT; ++i) {
        noc_async_read_page(i, s, wr);
        wr += tile_bytes;
    }
    if constexpr (fill2) {
        cb_reserve_back(cb_id2, NT);
        uint32_t wr2 = get_write_ptr(cb_id2);
        for (uint32_t i = 0; i < NT; ++i) {
            noc_async_read_page(i, s, wr2);
            wr2 += tile_bytes;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, NT);
    if constexpr (fill2) {
        cb_push_back(cb_id2, NT);
    }
}
