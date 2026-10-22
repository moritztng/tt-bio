// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_ln reader: gamma and beta once (Ct full tiles each, every row the weights), the reduce scaler,
// then the Ct channel tiles of each of this core's row blocks (32 rows, contiguous pages) into cb_in.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t num = get_arg_val<uint32_t>(1);

    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t SCALER = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = 0, cb_gamma = 2, cb_beta = 3, cb_scaler = 4;
    constexpr auto x_args = TensorAccessorArgs<2>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, x_addr);
    const uint32_t tb = get_tile_size(cb_in);

    generate_reduce_scaler(cb_scaler, SCALER);
    {
        const auto sg = TensorAccessor(g_args, gamma_addr);
        const auto sb = TensorAccessor(b_args, beta_addr);
        cb_reserve_back(cb_gamma, Ct);
        cb_reserve_back(cb_beta, Ct);
        uint32_t wg = get_write_ptr(cb_gamma), wb = get_write_ptr(cb_beta);
        for (uint32_t t = 0; t < Ct; ++t) {
            noc_async_read_page(t, sg, wg);
            noc_async_read_page(t, sb, wb);
            wg += tb;
            wb += tb;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Ct);
        cb_push_back(cb_beta, Ct);
    }
    for (uint32_t u = first; u < first + num; ++u) {
        cb_reserve_back(cb_in, Ct);
        uint32_t w = get_write_ptr(cb_in);
        for (uint32_t t = 0; t < Ct; ++t) {
            noc_async_read_page(u * Ct + t, sx, w);
            w += tb;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, Ct);
    }
}
