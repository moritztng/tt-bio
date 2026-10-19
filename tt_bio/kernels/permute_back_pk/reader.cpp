// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// permute_back_pk: x [C, N, N] channel-major -> out [N, N, H] channel-last, the re-tiling done by the packer.
// Per unit (ct, it, jt): the 32 channel tiles x[ct*32 + c][it, jt], in channel order, in groups of 8.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t ABL = get_compile_time_arg_val(1);   // bit 0: no DRAM read (diagnostic)
    constexpr auto x_args = TensorAccessorArgs<2>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t SS = St * St;
    const uint32_t tb = get_tile_size(in_cb);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t ct = u / SS;
        const uint32_t cube = u % SS;          // it * St + jt
        uint32_t page = ct * 32 * SS + cube;
        for (uint32_t g = 0; g < 4; ++g) {
            cb_reserve_back(in_cb, 8);
            uint32_t dst = get_write_ptr(in_cb);
            for (uint32_t c = 0; c < 8; ++c, page += SS, dst += tb)
                if constexpr (!(ABL & 1)) noc_async_read_page(page, x, dst);
            noc_async_read_barrier();
            cb_push_back(in_cb, 8);
        }
    }
}
