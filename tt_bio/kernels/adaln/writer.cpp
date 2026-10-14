// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// adaln writer (tt_bio/adaln.py): row tile r's Kt output tiles to pages r * Kt ..
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    const uint32_t tb = get_tile_size(out_cb);
    for (uint32_t r = r0; r < r0 + nrows; ++r) {
        cb_wait_front(out_cb, Kt);
        uint32_t src = get_read_ptr(out_cb);
        for (uint32_t j = 0; j < Kt; ++j, src += tb) noc_async_write_page(r * Kt + j, o, src);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, Kt);
    }
    noc_async_write_barrier();
}
