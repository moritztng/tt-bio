// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DiffDock edge-tile writer: per edge tile, the group's NOUT fp32 output tiles into columns
// OUT0.. of the [E, OUTW*32] output.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NOUT = get_compile_time_arg_val(0);
    constexpr uint32_t OUT0 = get_compile_time_arg_val(1);
    constexpr uint32_t OUTW = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t n_et = get_arg_val<uint32_t>(1);
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    constexpr uint32_t cb_out = 16;
    const uint32_t tb = get_tile_size(cb_out);
    for (uint32_t et = first; et < first + n_et; ++et) {
        cb_wait_front(cb_out, NOUT);
        uint32_t r = get_read_ptr(cb_out);
        for (uint32_t o = 0; o < NOUT; ++o, r += tb) noc_async_write_page(et * OUTW + OUT0 + o, so, r);
        noc_async_write_barrier();
        cb_pop_front(cb_out, NOUT);
    }
}
