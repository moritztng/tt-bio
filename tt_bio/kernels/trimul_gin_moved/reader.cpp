// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul gated in-projection with the channel move fused in. A unit is one (b, xt, yt) tile cube of
// the pair grid: the 32 row tiles R = (b, x = xt*32 + il, y-tile yt), il = 0..31, each Kt tiles of
// the activation. They go to c_1 at slot dt * 32 + il so that, for one k step, the 32 tiles of
// consecutive il are contiguous: compute reads them as the ct columns of its matmul block.
// Both transposed weights [Wp^T | Wg^T] (each 2*CT row tiles x Kt) are loaded into c_0 once.
// With MASK each unit also gets 32 row-broadcast mask tiles in c_6: tile il carries
// m[b, xt*32 + il, yt*32 + 0..31] in its row 0, cut out of the one mask tile (b, xt, yt).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t CT2 = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t MASK = get_compile_time_arg_val(3);
    constexpr uint32_t ABL = get_compile_time_arg_val(4);   // bit 3: no x reads (diagnostic)
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto wp_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wg_args = TensorAccessorArgs<wp_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
    // SPLIT (writer.cpp): after each unit's activations this RISC gathers rows 16..31 of every output
    // tile of the unit, NB tiles per read barrier, and writes them as the second 1 KB of each page.
    constexpr uint32_t SO = m_args.next_compile_time_args_offset();
    constexpr uint32_t SPLIT = get_compile_time_arg_val(SO);
    constexpr uint32_t NB = get_compile_time_arg_val(SO + 1);
    constexpr uint32_t CTO = get_compile_time_arg_val(SO + 2);
    constexpr uint32_t CTT = get_compile_time_arg_val(SO + 3);
    constexpr auto a_args = TensorAccessorArgs<SO + 4>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto wp = TensorAccessor(wp_args, get_common_arg_val<uint32_t>(1));
    const auto wg = TensorAccessor(wg_args, get_common_arg_val<uint32_t>(2));
    const auto mk = TensorAccessor(m_args, get_common_arg_val<uint32_t>(3));
    const auto oa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(4));
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(5));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t x_cb = tt::CBIndex::c_1;
    constexpr uint32_t mscratch_cb = tt::CBIndex::c_3;
    constexpr uint32_t m_cb = tt::CBIndex::c_6;
    constexpr uint32_t S = St * 32;
    const uint32_t tb = get_tile_size(x_cb);
    constexpr uint32_t stage_cb = tt::CBIndex::c_25;
    constexpr uint32_t CT = CT2 / 2, SS = St * St, TB = 2048, FB = 512, RB = 32;
    uint32_t stage0 = 0, seq = 0;
    bool dirty[2] = {false, false};
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    volatile tt_l1_ptr uint32_t* addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    if constexpr (SPLIT) {
        cb_reserve_back(stage_cb, 2 * NB);
        stage0 = get_write_ptr(stage_cb);
    }

    cb_reserve_back(w_cb, 2 * CT2 * Kt);
    uint32_t w = get_write_ptr(w_cb);
    for (uint32_t t = 0; t < CT2 * Kt; ++t, w += tb) noc_async_read_page(t, wp, w);
    for (uint32_t t = 0; t < CT2 * Kt; ++t, w += tb) noc_async_read_page(t, wg, w);
    noc_async_read_barrier();
    cb_push_back(w_cb, 2 * CT2 * Kt);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / (St * St);
        const uint32_t xt = (u / St) % St;
        const uint32_t yt = u % St;
        cb_reserve_back(x_cb, 32 * Kt);
        const uint32_t base = get_write_ptr(x_cb);
        uint32_t page = ((b * S + xt * 32) * St + yt) * Kt;
        for (uint32_t il = 0; il < 32; ++il, page += St * Kt) {
            uint32_t dst = base + il * tb;
            if constexpr (!(ABL & 8))
                for (uint32_t dt = 0; dt < Kt; ++dt, dst += 32 * tb) noc_async_read_page(page + dt, x, dst);
        }
        noc_async_read_barrier();
        cb_push_back(x_cb, 32 * Kt);
        if constexpr (MASK) {
            const uint32_t scratch = get_write_ptr(mscratch_cb);
            noc_async_read_page((b * St + xt) * St + yt, mk, scratch);
            noc_async_read_barrier();
            cb_reserve_back(m_cb, 32);
            uint32_t d = get_write_ptr(m_cb);
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            // Row il of a bf16 tile: 16 elements in face (il / 16) * 2 and 16 in the face after it,
            // row il % 16 of each; row 0 of the destination is the first row of faces 0 and 1.
            for (uint32_t il = 0; il < 32; ++il, d += tb) {
                volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d);
                const uint32_t s0 = (((il >> 4) * 2) * 256 + (il & 15) * 16) / 2;
                for (uint32_t k = 0; k < 8; ++k) {
                    dst[k] = src[s0 + k];
                    dst[128 + k] = src[s0 + 128 + k];
                }
            }
            cb_push_back(m_cb, 32);
        }
        if constexpr (SPLIT) {
            const uint32_t cube = u % SS;
            for (uint32_t q = 0; q < CT2; ++q) {
                ++seq;
                while (*go != seq) {
                    invalidate_l1_cache();
                }
                const uint32_t src = *addr;
                const bool is_a = q < CT;
                uint32_t page = (b * CTT + CTO + (is_a ? q : q - CT)) * 32 * SS + cube;
                // The one-packet read state is set per group: the mask copy above may have used the NOC.
                noc_async_read_one_packet_set_state(get_noc_addr(stage0), RB);
                for (uint32_t c0 = 0; c0 < 32 && !(ABL & 4); c0 += NB) {
                    const uint32_t grp = (c0 / NB) & 1u;
                    const uint32_t gbase = stage0 + grp * NB * TB;
                    if (dirty[grp]) noc_async_writes_flushed();
                    for (uint32_t j = 0; j < NB; ++j) {
                        const uint32_t c = c0 + j;
                        const uint32_t st = gbase + j * TB;
                        uint32_t s0 = src + 16 * TB + (c >> 4) * 2 * FB + (c & 15) * RB;
                        uint32_t s1 = s0 + FB, d0 = st, d1 = st + FB;
                        for (uint32_t il = 0; il < 16; ++il, s0 += TB, s1 += TB, d0 += RB, d1 += RB) {
                            noc_async_read_one_packet_with_state(s0, d0);
                            noc_async_read_one_packet_with_state(s1, d1);
                        }
                    }
                    noc_async_read_barrier();
                    for (uint32_t j = 0; j < NB; ++j, page += SS)
                        noc_async_write(gbase + j * TB,
                                        (is_a ? oa.get_noc_addr(page) : ob.get_noc_addr(page)) + 2 * FB, 2 * FB);
                    dirty[grp] = true;
                }
                *done = seq;
            }
            noc_async_write_barrier();
            dirty[0] = dirty[1] = false;
        }
    }
}
