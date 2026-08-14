// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 1's adjoint, reader side. The forward contracts a band of z-planes into one W tile; the adjoint
// spreads one W tile back across the same band. So where the forward reads nplane tiles and writes one,
// this reads ONE and writes nplane -- the transaction count is the same and only the direction changes,
// which is what makes the two cost about the same.
//
// The masks are the same tiles the forward uses, unchanged: the adjoint of a weighted sum is the same
// weights applied to the scattered value. They are fixed for the direction and loaded once.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(1);
    constexpr uint32_t nplane = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr auto w_args = TensorAccessorArgs<4>();
    constexpr auto m_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t nblocks = get_arg_val<uint32_t>(2);
    const uint32_t page0 = get_arg_val<uint32_t>(3);
    const uint32_t npages = get_arg_val<uint32_t>(4);

    const auto w = TensorAccessor(w_args, w_addr, tile_bytes);
    const auto m = TensorAccessor(m_args, m_addr, tile_bytes);

    cb_reserve_back(cb_mask, nplane);
    uint32_t wm = get_write_ptr(cb_mask);
    for (uint32_t p = 0; p < nplane; ++p) {
        noc_async_read_page(p, m, wm);
        wm += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_mask, nplane);

    for (uint32_t b = 0; b < nblocks; ++b) {
        cb_reserve_back(cb_w, 1);
        uint32_t pg = page0 + b;
        if (pg >= npages) {
            pg -= npages;
        }
        noc_async_read_page(pg, w, get_write_ptr(cb_w));
        noc_async_read_barrier();
        cb_push_back(cb_w, 1);
    }
}
