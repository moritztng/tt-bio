// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// inproj_gated reader. The trimul's in-projection and its gated channel move as one program: this
// kernel feeds the LN'd pair x [1, N, N, K] and the role's transposed weights, the compute kernel
// forms p^T and g^T per (position row, channel tile) as W^T @ x^T, and the writer is the gated
// move's 32-row reblock. The [1, N, N, 4C] projection is never written.
//
// Once per core: the role's W^T tiles, p then g, (ct, kt) row-major with Kt1 = Kt + HAS_BIAS
// columns (the last carries the bias in column 0), and when HAS_BIAS the ones tile that turns the
// bias into one more K step. Per group: the 32 rows x Kt tiles of x at (it, jt), row-major.
//
// A group is (it, jt, sub) with sub fastest. `sub` picks CPG of the role's Ct channel tiles, so a
// group's x tiles are read S = Ct / CPG times per role; S trades x reads for groups (cores).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ones_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t p_off = get_common_arg_val<uint32_t>(3);  // value rows of W^T, in tiles
    const uint32_t g_off = get_common_arg_val<uint32_t>(4);  // gate rows of W^T, in tiles
    const uint32_t first_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);
    const uint32_t D1 = get_arg_val<uint32_t>(3);
    const uint32_t S = get_arg_val<uint32_t>(4);
    const uint32_t group_stride = get_arg_val<uint32_t>(5);
    const uint32_t group_wrap_hi = get_arg_val<uint32_t>(6);
    const uint32_t group_wrap_lo = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t Ct = get_compile_time_arg_val(4);
    constexpr uint32_t HAS_BIAS = get_compile_time_arg_val(5);
    constexpr uint32_t Kt1 = Kt + HAS_BIAS;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, x_addr);
    const auto sw = TensorAccessor(w_args, w_addr);
    const auto so = TensorAccessor(o_args, ones_addr);
    const uint32_t tile_bytes = get_tile_size(cb_x);

    {
        cb_reserve_back(cb_w, 2 * Ct * Kt1);
        uint32_t l1 = get_write_ptr(cb_w);
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t base = (half ? g_off : p_off) * Kt1;
            for (uint32_t t = 0; t < Ct * Kt1; ++t) {
                noc_async_read_page(base + t, sw, l1);
                l1 += tile_bytes;
            }
        }
        if constexpr (HAS_BIAS) {
            cb_reserve_back(cb_ones, 1);
            noc_async_read_page(0, so, get_write_ptr(cb_ones));
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, 2 * Ct * Kt1);
        if constexpr (HAS_BIAS) {
            cb_push_back(cb_ones, 1);
        }
    }

    const uint32_t NtS = Nt * S;
    uint32_t group = first_group;
    for (uint32_t gi = 0; gi < num_groups; ++gi) {
        const uint32_t it = group / NtS;
        const uint32_t jt = (group - it * NtS) / S;
        const uint32_t row_abs = it * TILE_HEIGHT;
        const uint32_t rows_valid = (row_abs + TILE_HEIGHT <= D1) ? TILE_HEIGHT : (D1 - row_abs);

        cb_reserve_back(cb_x, TILE_HEIGHT * Kt);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t il = 0; il < TILE_HEIGHT; ++il) {
            // Padding rows read row 0's tiles so the group stays a fixed size; the writer zeroes
            // those rows of the output.
            const uint32_t row = il < rows_valid ? row_abs + il : 0;
            const uint32_t page = (row * Nt + jt) * Kt;
            for (uint32_t k = 0; k < Kt; ++k) {
                noc_async_read_page(page + k, sx, l1);
                l1 += tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, TILE_HEIGHT * Kt);

        group += group_stride;
        if (group == group_wrap_hi) {
            group = group_wrap_lo;
        }
    }
}
