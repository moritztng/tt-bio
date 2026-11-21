// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail, weights resident: writes this core's output rows, one block of Mb x Nt tiles.
// With SPLIT = 2 the Nt columns go to two tensors of Nt / 2 columns each (the gated in-projection's
// a and b); SPLIT = 1 passes the same tensor twice. T2_RES32 (SPLIT 1): c_2 is the float32 sum, written to o0,
// and c_21 its bfloat16 copy, written to the same pages of o1.
// T2STATS (compute_ln.cpp): per row tile, compute's mean and rstd tiles (c_24, fp32, column 0) become one 256-byte page
// of the ROW_MAJOR float32 stats tensor [B, S, S/32, 64] (32 means, then 32 rstd), page = the row tile's index, the
// layout trimul_in_t1 T1STATS reads. Common runtime arg 2: its address. c_25 is the staging page.
#include "api/dataflow/dataflow_api.h"
#ifndef T2STATS
#define T2STATS 0
#endif

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t Mb = get_compile_time_arg_val(1);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(2);
    constexpr uint32_t Ns = Nt / SPLIT;
    // T2_ABL bit 32 (bench only): blocks are popped without writing them.
    constexpr bool NOWRITE = get_compile_time_arg_val(3) & 32;
    constexpr auto o0_args = TensorAccessorArgs<4>();
    constexpr auto o1_args = TensorAccessorArgs<o0_args.next_compile_time_args_offset()>();
    const auto o0 = TensorAccessor(o0_args, get_common_arg_val<uint32_t>(0));
    const auto o1 = TensorAccessor(o1_args, get_common_arg_val<uint32_t>(1));
#if T2STATS
    constexpr auto st_args = TensorAccessorArgs<o1_args.next_compile_time_args_offset()>();
    const auto st = TensorAccessor(st_args, get_common_arg_val<uint32_t>(2));
    constexpr uint32_t cb_st = tt::CBIndex::c_24, cb_sp = tt::CBIndex::c_25;
    cb_reserve_back(cb_sp, 1);   // the writer's own staging page: never pushed
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_sp));
#endif
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    const uint32_t tb = get_tile_size(out_cb);
#ifdef T2_ZW
    // T2's z read moved off the reader (reader_ln.cpp T2_ZW): z of block b + 1 is read into c_7 (two blocks deep)
    // while compute works on block b, from o0, the tensor this writer then writes in place. Block b + 1's rows are
    // not block b's, so the read never sees a written row.
    constexpr uint32_t z_cb = tt::CBIndex::c_7;
    const uint32_t tz = get_tile_size(z_cb);
    auto read_z = [&](uint32_t m) {
        cb_reserve_back(z_cb, Mb * Nt);
        uint32_t w = get_write_ptr(z_cb);
        for (uint32_t t = 0; t < Mb * Nt; ++t, w += tz) noc_async_read_page(m * Nt + t, o0, w);
    };
    if (nblocks) {
        read_z(m0);
        noc_async_read_barrier();
        cb_push_back(z_cb, Mb * Nt);
    }
#endif
    for (uint32_t b = 0; b < nblocks; ++b) {
        const uint32_t m = m0 + b * Mb;
#ifdef T2_ZW
        const bool more = b + 1 < nblocks;
        if (more) read_z(m + Mb);
#endif
        cb_wait_front(out_cb, Mb * Nt);
        uint32_t r = get_read_ptr(out_cb);
        for (uint32_t i = 0; i < Mb; ++i) {
            const uint32_t row = (m + i) * Ns;
            if constexpr (NOWRITE) continue;
            for (uint32_t c = 0; c < Ns; ++c, r += tb) noc_async_write_page(row + c, o0, r);
            if constexpr (SPLIT == 2) {
                for (uint32_t c = 0; c < Ns; ++c, r += tb) noc_async_write_page(row + c, o1, r);
            }
        }
#ifdef T2_RES32
        constexpr uint32_t copy_cb = tt::CBIndex::c_21;
        const uint32_t tc = get_tile_size(copy_cb);
        cb_wait_front(copy_cb, Mb * Nt);
        uint32_t rc = get_read_ptr(copy_cb);
        for (uint32_t t = 0; t < (NOWRITE ? 0 : Mb * Nt); ++t, rc += tc) noc_async_write_page(m * Nt + t, o1, rc);
#endif
#ifdef T2_ZW
        if (more) {
            noc_async_read_barrier();
            cb_push_back(z_cb, Mb * Nt);
        }
#endif
        noc_async_writes_flushed();
        cb_pop_front(out_cb, Mb * Nt);
#if T2STATS
        for (uint32_t i = 0; i < Mb; ++i) {
            cb_wait_front(cb_st, 2);
            volatile tt_l1_ptr uint32_t* mn = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_st));
            volatile tt_l1_ptr uint32_t* rs = mn + 1024;
            // column 0 of a 32 x 32 fp32 tile: rows 0..15 in face 0, rows 16..31 in face 2, 16 values a face row
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t at = (row >> 4) * 512 + (row & 15) * 16;
                page[row] = mn[at];
                page[32 + row] = rs[at];
            }
            if constexpr (!NOWRITE) {
                noc_async_write_page(m + i, st, reinterpret_cast<uint32_t>(page));
                noc_async_write_barrier();
            }
            cb_pop_front(cb_st, 2);
        }
#endif
#ifdef T2_RES32
        cb_pop_front(copy_cb, Mb * Nt);
#endif
    }
    noc_async_write_barrier();
}
