// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// permute_fwd_pk: the 32 output tiles of unit (ct, it, jt), out[ct*32 + c][it, jt] in channel order, WB per flush.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t WB = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t SS = St * St;
    const uint32_t tb = get_tile_size(out_cb);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t ct = u % CT, cube = u / CT;
        uint32_t page = ct * 32 * SS + cube;
        for (uint32_t c0 = 0; c0 < 32; c0 += WB) {
            cb_wait_front(out_cb, WB);
            uint32_t src = get_read_ptr(out_cb);
            for (uint32_t k = 0; k < WB; ++k, page += SS, src += tb) noc_async_write_page(page, o, src);
            noc_async_writes_flushed();
            cb_pop_front(out_cb, WB);
        }
    }
    noc_async_write_barrier();
}
