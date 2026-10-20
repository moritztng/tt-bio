// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul gated in-projection with the channel move fused in: writes a and b as [B, C, S, S]. For
// each unit (b, xt, yt) and channel tile q, compute hands over 32 tiles in il order, tile il being
// [channel c, y] of row x = xt*32 + il. The output tile of channel c at (xt, yt) has row il = row c
// of tile il: 64 local face-row reads assemble it in a staging slot, then one 2 KB DRAM write. The
// same gather as reblock_permute's writer, whose comment carries the alignment argument.
//
// O8 (lever trimul_gin_o8): compute packs bfp8_b, whose tile is 64 exponent bytes (16 per face, one per 16-value
// face row) then the faces as 16 B mantissa rows. The gather moves whole face rows, so every shared exponent stays
// valid: 64 NOC reads of 16 B mantissa rows and a byte copy of the 64 exponents, then one 1088 B write.
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
    constexpr bool O8 = get_compile_time_arg_val(6);
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
    // Tile, face and face-row bytes, and where the face data starts (after bfp8_b's 64 exponent bytes).
    constexpr uint32_t TB = O8 ? 1088 : 2048, FB = O8 ? 256 : 512, RB = O8 ? 16 : 32, EB = O8 ? 64 : 0;

    // 2 * NB stage tiles: group g gathers NB tiles into its half while the other half's writes drain.
    cb_reserve_back(stage_cb, 2 * NB);
    const uint32_t stage0 = get_write_ptr(stage_cb);
    noc_async_read_one_packet_set_state(get_noc_addr(stage0), RB);
    bool dirty[2] = {false, false};

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / SS;
        const uint32_t cube = u % SS;          // xt * St + yt
        for (uint32_t q = 0; q < CT2; ++q) {
            const bool is_a = q < CT;
            uint32_t page = (b * CTT + CTO + (is_a ? q : q - CT)) * 32 * SS + cube;
            cb_wait_front(out_cb, 32);
            const uint32_t src = get_read_ptr(out_cb);
            for (uint32_t c0 = 0; c0 < 32 && !(ABL & 4); c0 += NB) {
                const uint32_t grp = (c0 / NB) & 1u;
                const uint32_t gbase = stage0 + grp * NB * TB;
                if (dirty[grp]) noc_async_writes_flushed();
                for (uint32_t j = 0; j < NB; ++j) {
                    const uint32_t c = c0 + j;
                    const uint32_t st = gbase + j * TB;
                    const uint32_t sc = src + EB + (c >> 4) * 2 * FB + (c & 15) * RB;
                    uint32_t s0 = sc, s1 = sc + FB, d0 = st + EB, d1 = st + EB + FB;
                    for (uint32_t il = 0; il < 16; ++il, s0 += TB, s1 += TB, d0 += RB, d1 += RB) {
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s1, d1);
                    }
                    d0 = st + EB + 2 * FB;
                    d1 = st + EB + 3 * FB;
                    for (uint32_t il = 0; il < 16; ++il, s0 += TB, s1 += TB, d0 += RB, d1 += RB) {
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s1, d1);
                    }
                    if constexpr (O8) {
                        // Exponent of row c, face (c >> 4) * 2 + h of tile il -> row il & 15, face (il >> 4) * 2 + h.
                        const volatile uint8_t* se = reinterpret_cast<const volatile uint8_t*>(src) + (c >> 4) * 32 + (c & 15);
                        volatile uint8_t* de = reinterpret_cast<volatile uint8_t*>(st);
                        for (uint32_t il = 0; il < 32; ++il, se += TB) {
                            const uint32_t o = (il >> 4) * 32 + (il & 15);
                            de[o] = se[0];
                            de[o + 16] = se[16];
                        }
                    }
                }
                noc_async_read_barrier();
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
