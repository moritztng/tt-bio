// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// outer_residual writer: c_16 (float32 sum) and c_17 (its bfloat16 copy) over this core's work items, each to z's
// tile (b, i, jt, ct) as the reader maps it. Common runtime args: the two output addresses. Per-core: first work
// item, count.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr uint32_t N = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<4>();
    constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const auto sl = TensorAccessor(l_args, get_common_arg_val<uint32_t>(1));
    const uint32_t p0 = get_arg_val<uint32_t>(0);
    const uint32_t np = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16, cb_lo = 17;
    constexpr uint32_t RUN = N * KT;
    const uint32_t os = get_tile_size(cb_out), ls = get_tile_size(cb_lo);
    for (uint32_t p = p0; p < p0 + np; p += GRAN) {
        const uint32_t n = p0 + np - p < GRAN ? p0 + np - p : GRAN;
        cb_wait_front(cb_out, GRAN);
        cb_wait_front(cb_lo, GRAN);
        uint32_t r = get_read_ptr(cb_out), rl = get_read_ptr(cb_lo);
        for (uint32_t k = 0; k < n; ++k, r += os, rl += ls) {
            const uint32_t w = p + k;
            const uint32_t bj = w / RUN;
            const uint32_t q = (((bj / NT) * N + (w / KT) % N) * NT + bj % NT) * KT + w % KT;
            noc_async_write_page(q, so, r);
            noc_async_write_page(q, sl, rl);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, GRAN);
        cb_pop_front(cb_lo, GRAN);
    }
}
