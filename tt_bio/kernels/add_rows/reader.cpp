// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_rows reader: tile t of this core's range reads z's page z_off + t into c_0 and the block's
// page t into c_1. One row block of a [1, S, S, C] pair is one contiguous page range, so z_off is
// the block's first page and nothing else is strided.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t z_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t z_off = get_common_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);

    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr auto z_args = TensorAccessorArgs<2>();
    constexpr auto b_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    const auto z = TensorAccessor(z_args, z_addr);
    const auto b = TensorAccessor(b_args, b_addr);
    constexpr uint32_t cb_z = 0, cb_b = 1;

    for (uint32_t i = 0; i < n; i += GRAN) {
        const uint32_t k_n = (n - i < GRAN) ? (n - i) : GRAN;
        cb_reserve_back(cb_z, k_n);
        cb_reserve_back(cb_b, k_n);
        uint32_t lz = get_write_ptr(cb_z), lb = get_write_ptr(cb_b);
        for (uint32_t k = 0; k < k_n; ++k, lz += page, lb += page) {
            noc_async_read(z.get_noc_addr(z_off + t0 + i + k), lz, page);
            noc_async_read(b.get_noc_addr(t0 + i + k), lb, page);
        }
        noc_async_read_barrier();
        cb_push_back(cb_z, k_n);
        cb_push_back(cb_b, k_n);
    }
}
