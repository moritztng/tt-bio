// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_rows writer: the sum of tile t goes back to z's page z_off + t, the page the reader took it
// from. Each page belongs to one core and is read before it is written, so the update is in place.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t z_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t z_off = get_common_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);

    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr auto z_args = TensorAccessorArgs<2>();
    const auto z = TensorAccessor(z_args, z_addr);
    constexpr uint32_t cb_out = 16;

    for (uint32_t i = 0; i < n; i += GRAN) {
        const uint32_t k_n = (n - i < GRAN) ? (n - i) : GRAN;
        cb_wait_front(cb_out, k_n);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t k = 0; k < k_n; ++k, l1 += page) {
            noc_async_write(l1, z.get_noc_addr(z_off + t0 + i + k), page);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, k_n);
    }
    noc_async_write_barrier();
}
