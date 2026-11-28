// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_mix writer: a unit's two output tiles (MSA row 0, MSA row 1) to [2, n, H * 32]. The backward
// (BW) writes six, dq0 dq1 dk0 dk1 dv0 dv1, to the unit's places in [2, n, 3 * H * 32].
#include "api/dataflow/dataflow_api.h"

#include "unit.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t BW = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    const auto so = TensorAccessor(o_args, out_addr);
    const auto range = col_mix::units<o_args.next_compile_time_args_offset()>();

    const uint32_t tile = get_tile_size(cb_out);
    for (uint32_t u = range.first; u < range.first + range.num; ++u) {
        if constexpr (BW) {
            const uint32_t row = u / H, h = u % H;
            cb_wait_front(cb_out, 6);
            uint32_t r = get_read_ptr(cb_out);
            for (uint32_t part = 0; part < 3; ++part) {
                for (uint32_t s = 0; s < 2; ++s) {
                    noc_async_write(r, so.get_noc_addr((s * NT + row) * 3 * H + part * H + h), tile);
                    r += tile;
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, 6);
        } else {
            cb_wait_front(cb_out, 2);
            const uint32_t r = get_read_ptr(cb_out);
            noc_async_write(r, so.get_noc_addr(u), tile);
            noc_async_write(r + tile, so.get_noc_addr(NT * H + u), tile);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 2);
        }
    }
}
