// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1(i) reader -- feed the dense coordinate stage. Constants once, then six euler tiles and two
// lattice tiles per block. Everything is already dense, so this reader does no gathering and no
// address arithmetic; the gather reader is Phase 1(ii) and consumes the addresses this stage emits.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_e = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xy = get_compile_time_arg_val(1);
    constexpr uint32_t cb_c = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(4);
    constexpr auto e_args = TensorAccessorArgs<5>();
    constexpr auto xy_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<xy_args.next_compile_time_args_offset()>();

    const uint32_t e_addr = get_arg_val<uint32_t>(0);
    const uint32_t xy_addr = get_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_arg_val<uint32_t>(2);

    const auto se = TensorAccessor(e_args, e_addr, tile_bytes);
    const auto sx = TensorAccessor(xy_args, xy_addr, tile_bytes);
    const auto sc = TensorAccessor(c_args, c_addr, tile_bytes);

    cb_reserve_back(cb_c, 7);
    uint32_t p = get_write_ptr(cb_c);
    for (uint32_t i = 0; i < 7; ++i) {
        noc_async_read_page(i, sc, p);
        p += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, 7);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_reserve_back(cb_e, 6);
        p = get_write_ptr(cb_e);
        for (uint32_t i = 0; i < 6; ++i) {
            noc_async_read_page(b * 6 + i, se, p);
            p += tile_bytes;
        }
        cb_reserve_back(cb_xy, 2);
        p = get_write_ptr(cb_xy);
        for (uint32_t i = 0; i < 2; ++i) {
            noc_async_read_page(b * 2 + i, sx, p);
            p += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_e, 6);
        cb_push_back(cb_xy, 2);
    }
}
