// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// outer_residual writer: c_16 (float32 sum) and c_17 (its bfloat16 copy) over this core's work items, each to z's
// tile (b, i, jt, ct) as the reader maps it. Common runtime args: the two output addresses. Per-core: first work
// item, count. Under OR_BF16 c_16 alone (the bfloat16 sum) to the one output.
#include "api/dataflow/dataflow_api.h"

#ifndef OR_DM
#define OR_DM 0
#endif
// OR_STATS: per finished row tile, compute's mean and rstd tiles (c_29, fp32, column 0) become one 256-byte page of the
// ROW_MAJOR float32 stats tensor [B, n, n/32, 64] (32 means, then 32 rstd), the layout trimul_in_t1 T1STATS reads.
// Common runtime arg 2: its address. c_30 is the page's staging slot.
#ifndef OR_STATS
#define OR_STATS 0
#endif

void kernel_main() {
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr uint32_t N = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<4>();
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t p0 = get_arg_val<uint32_t>(0);
    const uint32_t np = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t RUN = N * KT;
    const uint32_t os = get_tile_size(cb_out);
#ifndef OR_BF16
    constexpr uint32_t cb_lo = 17;
    constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto sl = TensorAccessor(l_args, get_common_arg_val<uint32_t>(1));
    const uint32_t ls = get_tile_size(cb_lo);
#endif
#if OR_STATS
#ifdef OR_BF16
    constexpr auto t_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
#else
    constexpr auto t_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
#endif
    const auto st = TensorAccessor(t_args, get_common_arg_val<uint32_t>(2));
    constexpr uint32_t cb_st = 29, cb_sp = 30;
    cb_reserve_back(cb_sp, 1);   // the writer's own staging page: never pushed
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_sp));
#endif
    for (uint32_t p = p0; p < p0 + np; p += GRAN) {
        const uint32_t n = p0 + np - p < GRAN ? p0 + np - p : GRAN;
        cb_wait_front(cb_out, GRAN);
        uint32_t r = get_read_ptr(cb_out);
#ifndef OR_BF16
        cb_wait_front(cb_lo, GRAN);
        uint32_t rl = get_read_ptr(cb_lo);
#endif
        for (uint32_t k = 0; k < n; ++k, r += os) {
            const uint32_t w = p + k;
            const uint32_t bj = w / RUN;
            const uint32_t q = (((bj / NT) * N + (w / KT) % N) * NT + bj % NT) * KT + w % KT;
            if constexpr (!(OR_DM & 4)) noc_async_write_page(q, so, r);
#ifndef OR_BF16
            noc_async_write_page(q, sl, rl);
            rl += ls;
#endif
        }
        noc_async_write_barrier();
#if OR_STATS
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t w = p + k;
            if (w % KT != KT - 1) continue;
            const uint32_t bj = w / RUN;
            cb_wait_front(cb_st, 2);
            volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_st));
            volatile tt_l1_ptr uint32_t* r = m + 1024;
            // column 0 of a 32 x 32 fp32 tile: rows 0..15 in face 0, rows 16..31 in face 2, 16 values a face row
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t at = (row >> 4) * 512 + (row & 15) * 16;
                page[row] = m[at];
                page[32 + row] = r[at];
            }
            noc_async_write_page(((bj / NT) * N + (w / KT) % N) * NT + bj % NT, st, reinterpret_cast<uint32_t>(page));
            noc_async_write_barrier();
            cb_pop_front(cb_st, 2);
        }
#endif
        cb_pop_front(cb_out, GRAN);
#ifndef OR_BF16
        cb_pop_front(cb_lo, GRAN);
#endif
    }
}
