// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul gated in-projection with the channel move fused in: writes a and b as [B, C, S, S]. For
// each unit (b, xt, yt) and channel tile q, compute hands over 32 tiles in il order, tile il being
// [channel c, y] of row x = xt*32 + il. The output tile of channel c at (xt, yt) has row il = row c
// of tile il: 64 local face-row reads assemble it in a staging slot, then one 2 KB DRAM write. The
// same gather as reblock_permute's writer, whose comment carries the alignment argument.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CT2 = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t ABL = get_compile_time_arg_val(2);   // bit 2: no gather, no write (diagnostic)
    constexpr uint32_t NB = get_compile_time_arg_val(3);    // output tiles gathered per read barrier
    // This launch's first channel tile in `a` and `b`, and how many each holds: a call too wide for
    // L1 runs as several launches over channel slices that write into the same two tensors.
    constexpr uint32_t CTO = get_compile_time_arg_val(4);
    constexpr uint32_t CTT = get_compile_time_arg_val(5);
    // 1: the reader gathers rows 16..31 of every output tile (its half of the face-row reads and the
    // matching 1 KB of each 2 KB tile write). Semaphore 0 hands it each 32-tile group, 2 the group's
    // address; the group is popped once semaphore 1 says the reader's reads of it have landed.
    constexpr uint32_t SPLIT = get_compile_time_arg_val(6);
    constexpr auto a_args = TensorAccessorArgs<7>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto oa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0));
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t stage_cb = tt::CBIndex::c_24;
    constexpr uint32_t CT = CT2 / 2;
    constexpr uint32_t SS = St * St;
    constexpr uint32_t TB = 2048, FB = 512, RB = 32;

    // 2 * NB stage tiles: group g gathers NB tiles into its half while the other half's writes drain.
    cb_reserve_back(stage_cb, 2 * NB);
    const uint32_t stage0 = get_write_ptr(stage_cb);
    noc_async_read_one_packet_set_state(get_noc_addr(stage0), RB);
    bool dirty[2] = {false, false};
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    volatile tt_l1_ptr uint32_t* addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    uint32_t seq = 0;
    // Rows of each output tile this RISC gathers: all 32, or the first 16 when SPLIT.
    constexpr uint32_t HALVES = SPLIT ? 1 : 2;

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / SS;
        const uint32_t cube = u % SS;          // xt * St + yt
        for (uint32_t q = 0; q < CT2; ++q) {
            const bool is_a = q < CT;
            uint32_t page = (b * CTT + CTO + (is_a ? q : q - CT)) * 32 * SS + cube;
            cb_wait_front(out_cb, 32);
            const uint32_t src = get_read_ptr(out_cb);
            if constexpr (SPLIT) {
                *addr = src;
                *go = ++seq;
            }
            for (uint32_t c0 = 0; c0 < 32 && !(ABL & 4); c0 += NB) {
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
                    for (uint32_t il = 0; il < 16 && HALVES == 2; ++il, s0 += TB, s1 += TB, d0 += RB, d1 += RB) {
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s1, d1);
                    }
                }
                noc_async_read_barrier();
                for (uint32_t j = 0; j < NB; ++j, page += SS)
                    noc_async_write(gbase + j * TB, is_a ? oa.get_noc_addr(page) : ob.get_noc_addr(page),
                                    HALVES * 2 * FB);
                dirty[grp] = true;
            }
            noc_async_write_barrier();
            dirty[0] = dirty[1] = false;
            if constexpr (SPLIT) {
                while (*done != seq) {
                    invalidate_l1_cache();
                }
            }
            cb_pop_front(out_cb, 32);
        }
    }
}
