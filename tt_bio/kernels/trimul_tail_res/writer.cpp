// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail, weights resident: writes this core's output rows, one block of Mb x Nt tiles.
// With SPLIT = 2 the Nt columns go to two tensors of Nt / 2 columns each (the gated in-projection's
// a and b); SPLIT = 1 passes the same tensor twice.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t Mb = get_compile_time_arg_val(1);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(2);
    constexpr uint32_t Ns = Nt / SPLIT;
    constexpr auto o0_args = TensorAccessorArgs<3>();
    constexpr auto o1_args = TensorAccessorArgs<o0_args.next_compile_time_args_offset()>();
    const auto o0 = TensorAccessor(o0_args, get_common_arg_val<uint32_t>(0));
    const auto o1 = TensorAccessor(o1_args, get_common_arg_val<uint32_t>(1));
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    const uint32_t tb = get_tile_size(out_cb);
    for (uint32_t b = 0; b < nblocks; ++b) {
        const uint32_t m = m0 + b * Mb;
        cb_wait_front(out_cb, Mb * Nt);
        uint32_t r = get_read_ptr(out_cb);
        for (uint32_t i = 0; i < Mb; ++i) {
            const uint32_t row = (m + i) * Ns;
            for (uint32_t c = 0; c < Ns; ++c, r += tb) noc_async_write_page(row + c, o0, r);
            if constexpr (SPLIT == 2) {
                for (uint32_t c = 0; c < Ns; ++c, r += tb) noc_async_write_page(row + c, o1, r);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(out_cb, Mb * Nt);
    }
    noc_async_write_barrier();
}
