// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul_gin_moved's writer for a ragged pair grid: the same face-row gather into [B, C, S, S] (St x St
// tiles per channel), units (cube, part) as in the reader, and the padding zeroed before the write.
// The bias makes p * sigmoid(g) nonzero where x or y >= S, and the outgoing contraction runs over the
// padded tiles, so rows il >= S - 32 * xt of the last row tile and columns >= S - 32 * yt of the last
// column tile are written as 0.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CT2 = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t NB = get_compile_time_arg_val(3);    // output tiles gathered per read barrier
    constexpr uint32_t QS = get_compile_time_arg_val(4);
    constexpr auto a_args = TensorAccessorArgs<5>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto oa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0));
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t stage_cb = tt::CBIndex::c_24;
    constexpr uint32_t CT = CT2 / 2;
    constexpr uint32_t QN = CT2 / QS;
    constexpr uint32_t SS = St * St;
    constexpr uint32_t TB = 2048, FB = 512, RB = 32;
    constexpr uint32_t LAST = S - (St - 1) * 32;   // valid rows / columns in the last tile

    cb_reserve_back(stage_cb, 2 * NB);
    const uint32_t stage0 = get_write_ptr(stage_cb);
    noc_async_read_one_packet_set_state(get_noc_addr(stage0), RB);
    bool dirty[2] = {false, false};

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t cu = u / QS;
        const uint32_t qa = (u % QS) * QN;
        const uint32_t b = cu / SS;
        const uint32_t cube = cu % SS;          // xt * St + yt
        const uint32_t r0 = cube / St == St - 1 ? LAST : 32;   // first zero row
        const uint32_t j0 = cube % St == St - 1 ? LAST : 32;   // first zero column
        for (uint32_t q = qa; q < qa + QN; ++q) {
            const bool is_a = q < CT;
            uint32_t page = (b * CT + (is_a ? q : q - CT)) * 32 * SS + cube;
            cb_wait_front(out_cb, 32);
            const uint32_t src = get_read_ptr(out_cb);
            for (uint32_t c0 = 0; c0 < 32; c0 += NB) {
                const uint32_t grp = (c0 / NB) & 1u;
                const uint32_t gbase = stage0 + grp * NB * TB;
                if (dirty[grp]) noc_async_writes_flushed();
                for (uint32_t j = 0; j < NB; ++j) {
                    const uint32_t c = c0 + j;
                    const uint32_t st = gbase + j * TB;
                    const uint32_t sc = src + (c >> 4) * 2 * FB + (c & 15) * RB;
                    uint32_t s0 = sc, s1 = sc + FB, d0 = st, d1 = st + FB;
                    for (uint32_t il = 0; il < 16; ++il, s0 += TB, s1 += TB, d0 += RB, d1 += RB) {
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s1, d1);
                    }
                    d0 = st + 2 * FB;
                    d1 = st + 3 * FB;
                    for (uint32_t il = 0; il < 16; ++il, s0 += TB, s1 += TB, d0 += RB, d1 += RB) {
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s1, d1);
                    }
                }
                noc_async_read_barrier();
                if (r0 < 32 || j0 < 32) {
                    for (uint32_t j = 0; j < NB; ++j) {
                        // Row r of a bf16 tile: 16 elements at face (r / 16) * 2, row r % 16, and 16
                        // in the face after it.
                        volatile tt_l1_ptr uint16_t* t =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gbase + j * TB);
                        for (uint32_t r = 0; r < 32; ++r) {
                            volatile tt_l1_ptr uint16_t* f = t + (r >> 4) * 512 + (r & 15) * 16;
                            for (uint32_t cc = r < r0 ? j0 : 0; cc < 32; ++cc) f[(cc >> 4) * 256 + (cc & 15)] = 0;
                        }
                    }
                }
                for (uint32_t j = 0; j < NB; ++j, page += SS)
                    noc_async_write(gbase + j * TB, is_a ? oa.get_noc_addr(page) : ob.get_noc_addr(page), TB);
                dirty[grp] = true;
            }
            noc_async_write_barrier();
            dirty[0] = dirty[1] = false;
            cb_pop_front(out_cb, 32);
        }
    }
}
