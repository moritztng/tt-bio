// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_transpose reader. [S1, S2, C] TILE -> [S2, S1, C]. A unit is (Jt, It, ct): the 32 input
// tiles (i, Jt, ct) for i in tile-row It, which hold every element the 32 output tiles (j, It, ct)
// for j in tile-row Jt need. Read in i order into c_0, one unit at a time; the writer shuffles.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t S1t = get_compile_time_arg_val(0);
    constexpr uint32_t S2t = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr auto s_args = TensorAccessorArgs<3>();
    const auto s = TensorAccessor(s_args, src_addr);

    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t num = get_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_in);

    for (uint32_t u = first; u < first + num; ++u) {
        const uint32_t ct = u % Ct;
        const uint32_t It = (u / Ct) % S1t;
        const uint32_t Jt = u / (Ct * S1t);
        cb_reserve_back(cb_in, 32);
        uint32_t w = get_write_ptr(cb_in);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_read_page(((It * 32 + r) * S2t + Jt) * Ct + ct, s, w);
            w += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, 32);
    }
}
