// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail T2m writer (reader_moved.cpp's partner). A unit (it, jt, part) is R = 32 / P rows i,
// R / Mb blocks of Mb rows; the block's row tiles are m = i * St + jt, St apart. Per block this
// RISC reads what compute needs besides x, g (or xb under GATE, gt tiles per row) into c_5 / c_6 and
// z into c_7, then writes the previous block's output from c_2 into z in place. Reading block b
// before writing block b - 1 keeps the double-buffered inputs one block ahead of compute; z of a
// row is always read before this same core writes it, and no other core touches the row.
#include "api/dataflow/dataflow_api.h"

// Bench-only (trimul_tail.T2_DMABL, 0 in every model path): 4 skips the g / z reads, 8 the writes.
#ifndef T2_DMABL
#define T2_DMABL 0
#endif
// T2_ZR (trimul_tail.T2_ZR): reader_moved.cpp reads g / z / mask, this RISC only writes.
#ifndef T2_ZR
#define T2_ZR 0
#endif

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t Mb = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);
    constexpr uint32_t GATE = get_compile_time_arg_val(5);
    // compute_ln.cpp's EXTRA. 4: the block's Nt tiles of a [N, C] per-row dropout mask into c_20 (row
    // tile m = i * St + jt takes mask tiles jt * Nt.., the same for every row of a block). 8: the output
    // goes to `o`, not into z in place (a checkpointed forward keeps z as its saved input).
    constexpr uint32_t EXTRA = get_compile_time_arg_val(6);
    constexpr auto g_args = TensorAccessorArgs<7>();
    constexpr auto z_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr uint32_t R = 32 / P;
    constexpr uint32_t BPU = R / Mb;                        // blocks per unit
    constexpr uint32_t gt = GATE ? Kt : Nt;
    constexpr uint32_t g_cb = GATE ? tt::CBIndex::c_6 : tt::CBIndex::c_5;
    constexpr uint32_t z_cb = tt::CBIndex::c_7, out_cb = tt::CBIndex::c_2;

    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(0));
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(1));
    const auto mk = TensorAccessor(m_args, get_common_arg_val<uint32_t>(2));
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(3));
    constexpr uint32_t mask_cb = tt::CBIndex::c_20;
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(z_cb);      // z and the output share a format (bfp8 under pair_bfp8z)
    const uint32_t tg = get_tile_size(g_cb);      // bf16, or a bfp8 g_out (trimul_t1g8)

    // first row tile of block b of this core; its rows follow St apart
    auto block_m = [&](uint32_t b) -> uint32_t {
        const uint32_t u = first_unit + b / BPU;
        const uint32_t part = u % P, blk = u / P;
        const uint32_t it = blk / St, jt = blk - (blk / St) * St;
        const uint32_t i = it * 32 + part * R + (b % BPU) * Mb;
        return i * St + jt;
    };
    auto read_block = [&](uint32_t m) {
        cb_reserve_back(g_cb, Mb * gt);
        cb_reserve_back(z_cb, Mb * Nt);
        uint32_t wg = get_write_ptr(g_cb), wz = get_write_ptr(z_cb);
        if constexpr (EXTRA & 4) {
            cb_reserve_back(mask_cb, Nt);
            uint32_t wm = get_write_ptr(mask_cb);
            const uint32_t tm = get_tile_size(mask_cb);
            const uint32_t jt = m % St;
            for (uint32_t t = 0; t < Nt; ++t, wm += tm) noc_async_read_page(jt * Nt + t, mk, wm);
        }
        for (uint32_t r = 0; r < ((T2_DMABL & 4) ? 0 : Mb); ++r, m += St) {
            for (uint32_t t = 0; t < gt; ++t, wg += tg) noc_async_read_page(m * gt + t, g, wg);
            for (uint32_t t = 0; t < Nt; ++t, wz += tb) noc_async_read_page(m * Nt + t, z, wz);
        }
        noc_async_read_barrier();
        cb_push_back(g_cb, Mb * gt);
        cb_push_back(z_cb, Mb * Nt);
        if constexpr (EXTRA & 4) cb_push_back(mask_cb, Nt);
    };
    auto write_block = [&](uint32_t m) {
        cb_wait_front(out_cb, Mb * Nt);
        uint32_t r = get_read_ptr(out_cb);
        for (uint32_t k = 0; k < ((T2_DMABL & 8) ? 0 : Mb); ++k, m += St) {
            for (uint32_t t = 0; t < Nt; ++t, r += tb) {
                if constexpr (EXTRA & 8) noc_async_write_page(m * Nt + t, o, r);
                else noc_async_write_page(m * Nt + t, z, r);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(out_cb, Mb * Nt);
    };

    const uint32_t nblocks = num_units * BPU;
    if constexpr (T2_ZR) {
        for (uint32_t b = 0; b < nblocks; ++b) write_block(block_m(b));
    } else {
        for (uint32_t b = 0; b < nblocks; ++b) {
            read_block(block_m(b));
            if (b > 0) write_block(block_m(b - 1));
        }
        if (nblocks > 0) write_block(block_m(nblocks - 1));
    }
    noc_async_write_barrier();
}
