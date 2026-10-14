// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// addcmul_at reader: x, y, z into c_0, c_1, c_2 for `out = x + y * z`.
//
// The output is an [.., H, W] tile grid of WT tiles a row. Each operand is either that grid or a
// column window of a WIDER tensor: tile (r, c) of operand k is page `r * STRIDE_k + off_k + c`.
// STRIDE_k is compile-time (it is the operand's own tile width), off_k is a common runtime arg, so
// one cached program serves every window of the same wide tensor (the DiT reads 24 blocks' s-terms
// out of one linear's output this way, `openfold3_diffusion_transformer.py`).
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known device-wedge
// cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t x_off = get_common_arg_val<uint32_t>(1);
    const uint32_t y_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t y_off = get_common_arg_val<uint32_t>(3);
    const uint32_t z_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t z_off = get_common_arg_val<uint32_t>(5);
    const uint32_t first_tile = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t XS = get_compile_time_arg_val(2);
    constexpr uint32_t YS = get_compile_time_arg_val(3);
    constexpr uint32_t ZS = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, x_addr);
    const auto sy = TensorAccessor(y_args, y_addr);
    const auto sz = TensorAccessor(z_args, z_addr);

    constexpr uint32_t cb_x = 0, cb_y = 1, cb_z = 2;
    const uint32_t tile = get_tile_size(cb_x);

    uint32_t r = first_tile / WT, c = first_tile % WT;
    for (uint32_t i = 0; i < num_tiles; i += GRAN) {
        const uint32_t n = (num_tiles - i < GRAN) ? (num_tiles - i) : GRAN;
        cb_reserve_back(cb_x, n);
        cb_reserve_back(cb_y, n);
        cb_reserve_back(cb_z, n);
        uint32_t wx = get_write_ptr(cb_x), wy = get_write_ptr(cb_y), wz = get_write_ptr(cb_z);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_page(r * XS + x_off + c, sx, wx);
            noc_async_read_page(r * YS + y_off + c, sy, wy);
            noc_async_read_page(r * ZS + z_off + c, sz, wz);
            wx += tile;
            wy += tile;
            wz += tile;
            if (++c == WT) {
                c = 0;
                ++r;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, n);
        cb_push_back(cb_y, n);
        cb_push_back(cb_z, n);
    }
}
