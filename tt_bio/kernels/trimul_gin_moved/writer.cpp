// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul gated in-projection with the channel move fused in: writes a and b as [B, C, S, S]. For
// each unit (b, xt, yt) and channel tile q, compute hands over 32 tiles in il order, tile il being
// [channel c, y] of row x = xt*32 + il. The output tile of channel c at (xt, yt) has row il = row c
// of tile il: 64 local face-row reads assemble it in a staging slot, then one 2 KB DRAM write
// (../trimul_in_t1/gather.hpp, T1's gather at IL 32). GINSPLIT: this kernel moves the odd channel
// tiles and reader.cpp the even ones, each on its own NOC.
#include "api/dataflow/dataflow_api.h"
#include "../trimul_in_t1/gather.hpp"

#ifndef GINSPLIT
#define GINSPLIT 0
#endif

void kernel_main() {
    constexpr uint32_t CT2 = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t ABL = get_compile_time_arg_val(2);   // bit 2: no gather, no write (diagnostic)
    constexpr uint32_t NB = get_compile_time_arg_val(3);    // output tiles gathered per read barrier
    // This launch's first channel tile in `a` and `b`, and how many each holds: a call too wide for
    // L1 runs as several launches over channel slices that write into the same two tensors.
    constexpr uint32_t CTO = get_compile_time_arg_val(4);
    constexpr uint32_t CTT = get_compile_time_arg_val(5);
    constexpr auto a_args = TensorAccessorArgs<6>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto oa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0));
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t CT = CT2 / 2;
    constexpr uint32_t SS = St * St;
    t1_gather::Gather<NB, 32, (ABL & 4) ? 96 : 0, false> gather;
    gather.init(tt::CBIndex::c_24);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / SS;
        const uint32_t cube = u % SS;          // xt * St + yt
        for (uint32_t q = GINSPLIT; q < CT2; q += 1 + GINSPLIT) {
            const uint32_t page = (b * CTT + CTO + (q < CT ? q : q - CT)) * 32 * SS + cube;
            if (q < CT) gather.tile(tt::CBIndex::c_2, oa, page, 0, SS);
            else gather.tile(tt::CBIndex::c_2, ob, page, 0, SS);
        }
    }
    noc_async_write_barrier();
}
