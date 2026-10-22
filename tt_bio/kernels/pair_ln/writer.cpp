// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_ln writer: the Ct normalized tiles of each row block, to the same pages of `out`.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t num = get_arg_val<uint32_t>(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 16;
    constexpr auto d_args = TensorAccessorArgs<1>();
    const auto sd = TensorAccessor(d_args, dst_addr);
    const uint32_t tb = get_tile_size(cb_out);
    for (uint32_t u = first; u < first + num; ++u) {
        cb_wait_front(cb_out, Ct);
        uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < Ct; ++t) {
            noc_async_write_page(u * Ct + t, sd, rd);
            rd += tb;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Ct);
    }
}
