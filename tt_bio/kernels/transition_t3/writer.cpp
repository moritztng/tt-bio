// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Transition T3 writer: the Kt output tiles of each unit to the pages the unit was read from (the
// output has x's shape and layout; with add_to_input it IS x, and each core writes only rows it has
// already read).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t out_cb = tt::CBIndex::c_6;
    const uint32_t tb = get_tile_size(out_cb);

    uint32_t page = u0 * Kt;
    for (uint32_t u = 0; u < nunits; ++u) {
        cb_wait_front(out_cb, Kt);
        uint32_t r = get_read_ptr(out_cb);
        for (uint32_t dt = 0; dt < Kt; ++dt, ++page, r += tb) noc_async_write_page(page, o, r);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, Kt);
    }
    noc_async_write_barrier();
}
