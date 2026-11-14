// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// back_ln writer: the Ct normalized tiles of row block (i, jt) to `out` ([N, N, H] row-major tiles).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2);
    constexpr uint32_t R = 32 / P;
    constexpr uint32_t cb_out = 16;
    constexpr auto d_args = TensorAccessorArgs<3>();
    const auto sd = TensorAccessor(d_args, dst_addr);
    const uint32_t tb = get_tile_size(cb_out);
    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const uint32_t part = u % P, blk = u / P;
        const uint32_t it = blk / Nt, jt = blk % Nt;
        for (uint32_t il = 0; il < R; ++il) {
            const uint32_t page = ((it * 32 + part * R + il) * Nt + jt) * Ct;
            cb_wait_front(cb_out, Ct);
            uint32_t rd = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < Ct; ++t) { noc_async_write(rd, sd.get_noc_addr(page + t), tb); rd += tb; }
            noc_async_write_barrier();
            cb_pop_front(cb_out, Ct);
        }
    }
}
