// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gather_tiled reader: for each unit (row r of the index, 32-wide column block jt) fetch the 32 table rows the
// index names into one row-major block of CT tiles, every read in flight behind one barrier. Columns past the
// index's width take a zero row, the value ttnn's own tilize pads with.
// The table may hold COPIES identical replicas STRIDE pages apart (STRIDE coprime to the DRAM bank count): read k of
// unit u goes to replica (u + k) % COPIES, so an index that names one row over and over (RFD3's CA distance bins,
// where every pair past the top edge takes the last row) spreads over every bank instead of queueing on one.
// GT_FMA (the define is the compile-time arg offset of a's accessor, inv's follows): also read unit u's CT tiles of a
// and its one tile of inv for the compute's a + gathered * inv. GT_ROWLIST (offset of the list's accessor): the core
// walks a host list of units (common/row_list.hpp; common args 2-4 list address, n, core count, runtime arg 2 its
// ordinal) instead of a contiguous range.
#include "api/dataflow/dataflow_api.h"
#ifdef GT_ROWLIST
#include "../common/row_list.hpp"
#endif

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);    // index row width (logical columns of the output)
    constexpr uint32_t JT = get_compile_time_arg_val(1);   // tile columns of that width
    constexpr uint32_t CT = get_compile_time_arg_val(2);   // tiles across one table row
    constexpr uint32_t IDX_BYTES = get_compile_time_arg_val(3);   // one index row, W padded to 8 entries
    constexpr uint32_t COPIES = get_compile_time_arg_val(4);
    constexpr uint32_t STRIDE = get_compile_time_arg_val(5);   // table pages between replicas
    constexpr auto idx_args = TensorAccessorArgs<6>();
    constexpr auto tab_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_rm = 0, cb_idx = 1, cb_zero = 2;
    constexpr uint32_t ROW = CT * 32 * 2;

    const auto s_idx = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(0), IDX_BYTES);
    const auto s_tab = TensorAccessor(tab_args, get_common_arg_val<uint32_t>(1), ROW);
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
#ifdef GT_FMA
    constexpr uint32_t cb_a = 3, cb_inv = 4;
    constexpr auto a_args = TensorAccessorArgs<GT_FMA>();
    constexpr auto i_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto s_a = TensorAccessor(a_args, get_common_arg_val<uint32_t>(5));
    const auto s_inv = TensorAccessor(i_args, get_common_arg_val<uint32_t>(6));
    const uint32_t ta = get_tile_size(cb_a);
#endif

    const uint32_t zero_l1 = get_write_ptr(cb_zero);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
    for (uint32_t w = 0; w < ROW / 4; ++w) {
        z[w] = 0;
    }
    const uint64_t zero_noc = get_noc_addr(zero_l1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    uint32_t row = 0xFFFFFFFFu;
#ifdef GT_ROWLIST
    const uint32_t me = get_arg_val<uint32_t>(2);
    const auto list = row_list_read<GT_ROWLIST, GT_LIST_BYTES>(me, get_common_arg_val<uint32_t>(2), get_write_ptr(5));
    const uint32_t nl = row_list_count(me, get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4));
    for (uint32_t i = 0; i < nl; ++i) {
        const uint32_t u = list[i];
#else
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
#endif
        const uint32_t r = u / JT;
        const uint32_t jt = u % JT;
        if (r != row) {
            noc_async_read(s_idx.get_noc_addr(r), idx_l1, IDX_BYTES);
            noc_async_read_barrier();
            row = r;
        }
        const uint32_t c0 = jt * 32;
        const uint32_t n = (c0 + 32 <= W) ? 32 : W - c0;
        cb_reserve_back(cb_rm, CT);
        uint32_t wp = get_write_ptr(cb_rm);
        uint32_t rep = (u % COPIES) * STRIDE;
        for (uint32_t k = 0; k < 32; ++k, wp += ROW) {
            noc_async_read(k < n ? s_tab.get_noc_addr(rep + idx[c0 + k]) : zero_noc, wp, ROW);
            rep = rep + STRIDE == COPIES * STRIDE ? 0 : rep + STRIDE;
        }
#ifdef GT_FMA
        cb_reserve_back(cb_a, CT);
        cb_reserve_back(cb_inv, 1);
        uint32_t ap = get_write_ptr(cb_a);
        for (uint32_t t = 0; t < CT; ++t, ap += ta) {
            noc_async_read_page(u * CT + t, s_a, ap);
        }
        noc_async_read_page(u, s_inv, get_write_ptr(cb_inv));
#endif
        noc_async_read_barrier();
        cb_push_back(cb_rm, CT);
#ifdef GT_FMA
        cb_push_back(cb_a, CT);
        cb_push_back(cb_inv, 1);
#endif
    }
}
