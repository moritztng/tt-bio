// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail: writes each unit's Nt output tiles (b, jt, n) to page u * Nt + n of the
// output, which with RESID is z itself. In place is safe: a unit's z tiles are read by this core's
// reader before compute can produce them, and no other unit touches them.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    const uint32_t tb = get_tile_size(out_cb);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        cb_wait_front(out_cb, Nt);
        uint32_t rp = get_read_ptr(out_cb);
        for (uint32_t n = 0; n < Nt; ++n, rp += tb) noc_async_write_page(u * Nt + n, out, rp);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, Nt);
    }
    noc_async_write_barrier();
}
