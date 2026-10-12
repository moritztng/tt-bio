// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail, ending node (reader_tr.cpp): block (bt, At)'s unit q is z's row tile
// (b = bt*32 + q, At), written to pages (b * At_n + At) * Nt + n. In place over z is safe for the same
// reason as writer.cpp: the unit's z tiles were read by this core's reader before compute produced it.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t At_n = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));
    const uint32_t blk0 = get_arg_val<uint32_t>(0);
    const uint32_t nblk = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    const uint32_t tb = get_tile_size(out_cb);

    for (uint32_t blk = blk0; blk < blk0 + nblk; ++blk) {
        const uint32_t bt = blk / At_n;
        const uint32_t At = blk % At_n;
        for (uint32_t q = 0; q < 32; ++q) {
            const uint32_t u = (bt * 32 + q) * At_n + At;
            cb_wait_front(out_cb, Nt);
            uint32_t rp = get_read_ptr(out_cb);
            for (uint32_t n = 0; n < Nt; ++n, rp += tb) noc_async_write_page(u * Nt + n, out, rp);
            noc_async_writes_flushed();
            cb_pop_front(out_cb, Nt);
        }
    }
    noc_async_write_barrier();
}
