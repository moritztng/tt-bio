// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1(iii) reader -- feed the blend. Constants once, then eight gathered corner tiles and five
// dense per-pair tiles per block. In the assembled kernel the corners come from the gather reader's
// CB and the dense tiles from the coordinate stage, both already in L1; this arm reads them from
// DRAM so the blend can be graded on its own.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_slot = get_compile_time_arg_val(0);
    constexpr uint32_t cb_den = get_compile_time_arg_val(1);
    constexpr uint32_t cb_c = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(4);
    constexpr auto s_args = TensorAccessorArgs<5>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();

    const auto ss = TensorAccessor(s_args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto sd = TensorAccessor(d_args, get_arg_val<uint32_t>(1), tile_bytes);
    const auto sc = TensorAccessor(c_args, get_arg_val<uint32_t>(2), tile_bytes);

    cb_reserve_back(cb_c, 2);
    uint32_t p = get_write_ptr(cb_c);
    for (uint32_t i = 0; i < 2; ++i) {
        noc_async_read_page(i, sc, p);
        p += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, 2);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_reserve_back(cb_slot, 8);
        p = get_write_ptr(cb_slot);
        for (uint32_t i = 0; i < 8; ++i) {
            noc_async_read_page(b * 8 + i, ss, p);
            p += tile_bytes;
        }
        cb_reserve_back(cb_den, 5);
        p = get_write_ptr(cb_den);
        for (uint32_t i = 0; i < 5; ++i) {
            noc_async_read_page(b * 5 + i, sd, p);
            p += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_slot, 8);
        cb_push_back(cb_den, 5);
    }
}
