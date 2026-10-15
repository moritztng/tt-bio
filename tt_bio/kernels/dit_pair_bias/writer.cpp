// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dit_pair_bias writer. Per unit (b, ib, jb) and head tile hb, compute hands over the 32 tiles of heads
// hb * 32 .. + 31 in head order; head h goes to page ((b * H + h) * NT + ib) * NT + jb of [B, H, n, n].
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t HT = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t SW = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16;
    const uint32_t ts = get_tile_size(cb_out);
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / (NT * NT);
        const uint32_t ib = (u / NT) % NT;
        const uint32_t jb = u % NT;
        for (uint32_t h0 = 0; h0 < HT * 32; h0 += SW) {
            cb_wait_front(cb_out, SW);
            uint32_t rd = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < SW; ++t, rd += ts) {
                noc_async_write_page(((b * HT * 32 + h0 + t) * NT + ib) * NT + jb, so, rd);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, SW);
        }
    }
    noc_async_write_barrier();
}
