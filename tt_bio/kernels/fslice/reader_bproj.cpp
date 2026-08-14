// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Backprojection reader. The adjoint's INPUT is a slice tile, so this is a plain bulk tile read -- the
// per-row offsets that dominate the forward reader move to the writer side in the adjoint, exactly as
// section 5 said. It also loads the three coefficient tiles and the three transposed selection
// matrices once, since both are fixed for the orientation.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_y = get_compile_time_arg_val(0);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(1);
    constexpr uint32_t cb_selt = get_compile_time_arg_val(2);
    constexpr uint32_t nselt = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr auto y_args = TensorAccessorArgs<5>();
    constexpr auto c_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);
    const uint32_t ntiles = get_arg_val<uint32_t>(3);
    const uint32_t page0 = get_arg_val<uint32_t>(4);
    const uint32_t npages = get_arg_val<uint32_t>(5);

    const auto y = TensorAccessor(y_args, y_addr, tile_bytes);
    const auto cc = TensorAccessor(c_args, c_addr, tile_bytes);
    const auto ss = TensorAccessor(s_args, s_addr, tile_bytes);

    cb_reserve_back(cb_coef, 3);
    uint32_t w = get_write_ptr(cb_coef);
    for (uint32_t d = 0; d < 3; ++d) {
        noc_async_read_page(d, cc, w);
        w += tile_bytes;
    }
    cb_reserve_back(cb_selt, nselt);
    uint32_t ws = get_write_ptr(cb_selt);
    for (uint32_t d = 0; d < nselt; ++d) {
        noc_async_read_page(d, ss, ws);
        ws += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_coef, 3);
    cb_push_back(cb_selt, nselt);

    for (uint32_t t = 0; t < ntiles; ++t) {
        cb_reserve_back(cb_y, 1);
        uint32_t pg = page0 + t;
        if (pg >= npages) {
            pg -= npages;
        }
        noc_async_read_page(pg, y, get_write_ptr(cb_y));
        noc_async_read_barrier();
        cb_push_back(cb_y, 1);
    }
}
