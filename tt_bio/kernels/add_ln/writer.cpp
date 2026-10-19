// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_ln writer (tt_bio/add_ln.py): per row tile r of this core's range, the Kt tiles of h (c_2) and then of
// layer_norm(h) (c_3) to pages r * Kt .. of their outputs, four tiles per flush as compute streams them.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr auto h_args = TensorAccessorArgs<1>();
    constexpr auto y_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto h = TensorAccessor(h_args, get_common_arg_val<uint32_t>(0));
    const auto y = TensorAccessor(y_args, get_common_arg_val<uint32_t>(1));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);
    constexpr uint32_t h_cb = tt::CBIndex::c_2;
    constexpr uint32_t y_cb = tt::CBIndex::c_3;
    const uint32_t tb = get_tile_size(h_cb);

    constexpr uint32_t BLK = 4;
    auto drain = [&](uint32_t cb, const auto& dst, uint32_t page) {
        for (uint32_t i0 = 0; i0 < Kt; i0 += BLK) {
            const uint32_t n = Kt - i0 < BLK ? Kt - i0 : BLK;
            cb_wait_front(cb, n);
            uint32_t src = get_read_ptr(cb);
            for (uint32_t t = 0; t < n; ++t, src += tb) noc_async_write_page(page + i0 + t, dst, src);
            noc_async_writes_flushed();
            cb_pop_front(cb, n);
        }
    };
    for (uint32_t page = r0 * Kt, i = 0; i < nrows; ++i, page += Kt) {
        drain(h_cb, h, page);
        drain(y_cb, y, page);
    }
    noc_async_write_barrier();
}
