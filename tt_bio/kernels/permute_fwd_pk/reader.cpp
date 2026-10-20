// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// permute_fwd_pk: x [N, N, H] channel-last -> out [C, N, N] channel-major, the re-tiling done by the packer.
// Per unit (ct, it, jt): the 32 tiles x[it*32 + i][jt, CO + ct] ([j x c] each), in row order, in groups of 8.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t CO = get_compile_time_arg_val(2);
    constexpr uint32_t CT = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    const uint32_t tb = get_tile_size(in_cb);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t ct = u % CT, cube = u / CT;            // channel tile fastest: spreads the banks
        const uint32_t it = cube / St, jt = cube % St;
        uint32_t page = ((it * 32) * St + jt) * Ht + CO + ct;
        for (uint32_t g = 0; g < 4; ++g) {
            cb_reserve_back(in_cb, 8);
            uint32_t dst = get_write_ptr(in_cb);
            for (uint32_t i = 0; i < 8; ++i, page += St * Ht, dst += tb) noc_async_read_page(page, x, dst);
            noc_async_read_barrier();
            cb_push_back(in_cb, 8);
        }
    }
}
