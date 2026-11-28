// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_mix reader. qkv is [2, n, 3 * H * 32]: per MSA row and token tile row, H tiles of q, H of k,
// H of v. For each unit (tile row r, head h) it streams k0, k1, q0, q1, v0, v1 into c_0, and the
// tile row's logit-gap bias into c_1 when there is one. The two constant tiles (the head sum and
// the bias spread) go into c_2 once and stay. The backward (BW) also streams the unit's two cotangent
// tiles, d0 and d1, into c_3.
#include "api/dataflow/dataflow_api.h"

#include "unit.h"

void kernel_main() {
    const uint32_t qkv_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t const_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t bias_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t d_addr = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t cb_in = 0, cb_bias = 1, cb_const = 2, cb_d = 3;
    constexpr uint32_t NT = get_compile_time_arg_val(0);    // token tile rows
    constexpr uint32_t H = get_compile_time_arg_val(1);     // heads
    constexpr uint32_t BIAS = get_compile_time_arg_val(2);
    constexpr uint32_t BW = get_compile_time_arg_val(3);

    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto c_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, qkv_addr);
    const auto sc = TensorAccessor(c_args, const_addr);
    constexpr auto d_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto sb = TensorAccessor(b_args, bias_addr);
    const auto sd = TensorAccessor(d_args, d_addr);
    const auto range = col_mix::units<d_args.next_compile_time_args_offset()>();

    const uint32_t tile = get_tile_size(cb_in);
    cb_reserve_back(cb_const, 2);
    uint32_t w = get_write_ptr(cb_const);
    noc_async_read_page(0, sc, w);
    noc_async_read_page(1, sc, w + tile);
    noc_async_read_barrier();
    cb_push_back(cb_const, 2);

    constexpr uint32_t ROW = 3 * H;          // tiles a token tile row of one MSA row
    constexpr uint32_t SLAB = NT * ROW;      // tiles an MSA row
    for (uint32_t u = range.first; u < range.first + range.num; ++u) {
        const uint32_t r = u / H, h = u % H;
        cb_reserve_back(cb_in, 6);
        w = get_write_ptr(cb_in);
        for (uint32_t part : {H + h, h, 2 * H + h}) {       // k, q, v
            for (uint32_t s = 0; s < 2; ++s) {
                noc_async_read_page(s * SLAB + r * ROW + part, sx, w);
                w += tile;
            }
        }
        if constexpr (BIAS) {
            cb_reserve_back(cb_bias, 1);
            noc_async_read_page(r, sb, get_write_ptr(cb_bias));
        }
        if constexpr (BW) {
            cb_reserve_back(cb_d, 2);
            w = get_write_ptr(cb_d);
            noc_async_read_page(u, sd, w);
            noc_async_read_page(NT * H + u, sd, w + tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, 6);
        if constexpr (BIAS) {
            cb_push_back(cb_bias, 1);
        }
        if constexpr (BW) {
            cb_push_back(cb_d, 2);
        }
    }
}
