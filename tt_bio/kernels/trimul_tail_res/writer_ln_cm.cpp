// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail T2CM writer: z rows back in place, in the reader's cube order. Unit u = (it, jt) is the 32
// row tiles m = (it*32 + il)*St + jt; each block of Mb of them is Mb x Nt tiles in c_2, row tile m at
// pages m*Nt .. m*Nt + Nt - 1.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t Mb = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t TILE = 32;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    const uint32_t tb = get_tile_size(out_cb);
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t it = u / St;
        const uint32_t jt = u - it * St;
        for (uint32_t il0 = 0; il0 < TILE; il0 += Mb) {
            cb_wait_front(out_cb, Mb * Nt);
            uint32_t r = get_read_ptr(out_cb);
            for (uint32_t mi = 0; mi < Mb; ++mi) {
                const uint32_t row = ((it * TILE + il0 + mi) * St + jt) * Nt;
                for (uint32_t c = 0; c < Nt; ++c, r += tb) noc_async_write_page(row + c, o, r);
            }
            noc_async_writes_flushed();
            cb_pop_front(out_cb, Mb * Nt);
        }
    }
    noc_async_write_barrier();
}
