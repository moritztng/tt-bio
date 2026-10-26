// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail: writes each unit's Nt output tiles (b, jt, n) to page u * Nt + n of the
// output, which with RESID is z itself. With RESID this kernel also reads the unit's Nt z tiles into
// c_3 first: on its own NOC the z stream does not queue behind the reader's o and g reads (a bf16 gate
// at S 512 read 6.2 ms with z on the reader's NOC, 5.3 here). In place is safe: a unit's z tiles are
// read before compute can produce that unit's output, by the RISC that then writes it, and no other
// unit touches them.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t RESID = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t z_cb = tt::CBIndex::c_3;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    const uint32_t tb = get_tile_size(out_cb);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        if constexpr (RESID) {
            cb_reserve_back(z_cb, Nt);
            uint32_t zp = get_write_ptr(z_cb);
            for (uint32_t n = 0; n < Nt; ++n, zp += tb) noc_async_read_page(u * Nt + n, out, zp);
            noc_async_read_barrier();
            cb_push_back(z_cb, Nt);
        }
        cb_wait_front(out_cb, Nt);
        uint32_t rp = get_read_ptr(out_cb);
        for (uint32_t n = 0; n < Nt; ++n, rp += tb) noc_async_write_page(u * Nt + n, out, rp);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, Nt);
    }
    noc_async_write_barrier();
}
