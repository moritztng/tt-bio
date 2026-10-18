// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gather_tiled reader: for each unit (row r of the index, 32-wide column block jt) fetch the 32 table rows the
// index names into one row-major block of CT tiles, every read in flight behind one barrier. Columns past the
// index's width take a zero row, the value ttnn's own tilize pads with. With GT_FUSE it also reads the unit's
// scale tile (a [.., W, 1] TILE tensor: one tile per unit) and its CT shift tiles (same pages as the output).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);    // index row width (logical columns of the output)
    constexpr uint32_t JT = get_compile_time_arg_val(1);   // tile columns of that width
    constexpr uint32_t CT = get_compile_time_arg_val(2);   // tiles across one table row
    constexpr uint32_t IDX_BYTES = get_compile_time_arg_val(3);   // one index row, W padded to 8 entries
    constexpr auto idx_args = TensorAccessorArgs<4>();
    constexpr auto tab_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
#ifdef GT_FUSE
    constexpr auto sc_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();
    constexpr auto sh_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_sc = 3, cb_sh = 4;
    const uint32_t ts = get_tile_size(cb_sh);
    const auto s_sc = TensorAccessor(sc_args, get_common_arg_val<uint32_t>(2), ts);
    const auto s_sh = TensorAccessor(sh_args, get_common_arg_val<uint32_t>(3), ts);
#endif
    constexpr uint32_t cb_rm = 0, cb_idx = 1, cb_zero = 2;
    constexpr uint32_t ROW = CT * 32 * 2;

    const auto s_idx = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(0), IDX_BYTES);
    const auto s_tab = TensorAccessor(tab_args, get_common_arg_val<uint32_t>(1), ROW);
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    const uint32_t zero_l1 = get_write_ptr(cb_zero);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
    for (uint32_t w = 0; w < ROW / 4; ++w) {
        z[w] = 0;
    }
    const uint64_t zero_noc = get_noc_addr(zero_l1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    uint32_t row = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
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
        for (uint32_t k = 0; k < 32; ++k, wp += ROW) {
            noc_async_read(k < n ? s_tab.get_noc_addr(idx[c0 + k]) : zero_noc, wp, ROW);
        }
#ifdef GT_FUSE
        cb_reserve_back(cb_sc, 1);
        cb_reserve_back(cb_sh, CT);
        noc_async_read_page(u, s_sc, get_write_ptr(cb_sc));
        uint32_t sp = get_write_ptr(cb_sh);
        for (uint32_t t = 0; t < CT; ++t, sp += ts) {
            noc_async_read_page(u * CT + t, s_sh, sp);
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb_rm, CT);
#ifdef GT_FUSE
        cb_push_back(cb_sc, 1);
        cb_push_back(cb_sh, CT);
#endif
    }
}
