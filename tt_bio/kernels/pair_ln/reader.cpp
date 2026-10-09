// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pair layer norm reader: gamma and beta once (float32, rows already broadcast), the reduce
// scaler (1/K, exact: K is a power of two) and an eps tile; then Wt tiles of x per tile-row,
// the next row's reads issued while the compute kernel works on this one (CB depth 2 rows).
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t first_row = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t SCALER = get_compile_time_arg_val(1);   // packed bf16 pair
    constexpr uint32_t EPS = get_compile_time_arg_val(2);      // fp32 bits
    constexpr uint32_t cb_x = 0, cb_gamma = 1, cb_beta = 2, cb_scaler = 3, cb_eps = 4;
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, x_addr);

    generate_reduce_scaler(cb_scaler, SCALER);
    {
        cb_reserve_back(cb_eps, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(cb_eps, 1);
    }
    {
        const auto sg = TensorAccessor(g_args, gamma_addr);
        const auto sb = TensorAccessor(b_args, beta_addr);
        const uint32_t tb = get_tile_size(cb_gamma);
        cb_reserve_back(cb_gamma, Wt);
        cb_reserve_back(cb_beta, Wt);
        uint32_t wg = get_write_ptr(cb_gamma), wb = get_write_ptr(cb_beta);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_page(j, sg, wg);
            noc_async_read_page(j, sb, wb);
            wg += tb; wb += tb;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Wt);
        cb_push_back(cb_beta, Wt);
    }
    const uint32_t tx = get_tile_size(cb_x);
    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t page = (first_row + r) * Wt;
        cb_reserve_back(cb_x, Wt);
        uint32_t wx = get_write_ptr(cb_x);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_page(page + j, sx, wx);
            wx += tx;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, Wt);
    }
}
