// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 reader -- one reference tile, one shift-stack tile and one weight tile per pixel block.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_ref = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sh = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(4);
    constexpr auto r_args = TensorAccessorArgs<5>();
    constexpr auto s_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    const auto sr = TensorAccessor(r_args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto ss = TensorAccessor(s_args, get_arg_val<uint32_t>(1), tile_bytes);
    const auto sw = TensorAccessor(w_args, get_arg_val<uint32_t>(2), tile_bytes);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_reserve_back(cb_ref, 1);
        noc_async_read_page(b, sr, get_write_ptr(cb_ref));
        cb_reserve_back(cb_sh, 1);
        noc_async_read_page(b, ss, get_write_ptr(cb_sh));
        cb_reserve_back(cb_w, 1);
        noc_async_read_page(b, sw, get_write_ptr(cb_w));
        noc_async_read_barrier();
        cb_push_back(cb_ref, 1);
        cb_push_back(cb_sh, 1);
        cb_push_back(cb_w, 1);
    }
}
