// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_build writer: unit u's KT float32 tiles are pages u * KT .. u * KT + KT - 1 of z [B, N, N, C].
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr uint32_t cb_out = 16;
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    const uint32_t ts = get_tile_size(cb_out);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        cb_wait_front(cb_out, KT);
        uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < KT; ++t, rd += ts) {
            noc_async_write_page(u * KT + t, so, rd);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, KT);
    }
    noc_async_write_barrier();
}
