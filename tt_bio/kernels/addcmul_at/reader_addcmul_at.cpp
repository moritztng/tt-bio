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
// x and z may hold one sample's rows (XH / ZH row tiles) where y holds several (YH): their row is
// r % XH, so every diffusion sample reads the one conditioning row block.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known device-wedge
// cause in this codebase).
#include "api/dataflow/dataflow_api.h"

#include "../genq/genq_split.h"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t x_off = get_common_arg_val<uint32_t>(1);
    const uint32_t y_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t y_off = get_common_arg_val<uint32_t>(3);
    const uint32_t z_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t z_off = get_common_arg_val<uint32_t>(5);
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t XS = get_compile_time_arg_val(2);
    constexpr uint32_t YS = get_compile_time_arg_val(3);
    constexpr uint32_t ZS = get_compile_time_arg_val(4);
    constexpr uint32_t XH = get_compile_time_arg_val(5);
    constexpr uint32_t YH = get_compile_time_arg_val(6);
    constexpr uint32_t ZH = get_compile_time_arg_val(7);
    // genq compact split (tt_bio/genq.py): args 8..15, flag then seven constants.
    constexpr uint32_t G = 8;
    uint32_t first_tile, num_tiles;
    if constexpr (get_compile_time_arg_val(G) == 1) {
        const auto s = genq::slice<get_compile_time_arg_val(G + 1), get_compile_time_arg_val(G + 2),
                                   get_compile_time_arg_val(G + 3), get_compile_time_arg_val(G + 4),
                                   get_compile_time_arg_val(G + 5), get_compile_time_arg_val(G + 6),
                                   get_compile_time_arg_val(G + 7)>(
            get_absolute_logical_x(), get_absolute_logical_y());
        first_tile = s.first;
        num_tiles = s.num;
    } else {
        first_tile = get_arg_val<uint32_t>(0);
        num_tiles = get_arg_val<uint32_t>(1);
    }
    constexpr auto x_args = TensorAccessorArgs<G + 8>();
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
            const uint32_t rx = (XH == YH) ? r : r % XH;
            const uint32_t rz = (ZH == YH) ? r : r % ZH;
            noc_async_read_page(rx * XS + x_off + c, sx, wx);
            noc_async_read_page(r * YS + y_off + c, sy, wy);
            noc_async_read_page(rz * ZS + z_off + c, sz, wz);
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
