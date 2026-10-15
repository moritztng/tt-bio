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
// With SPLIT the reader also gathers and writes channels 16..31 of every channel tile the compute
// hands the writer (../trimul_in_t1/gather.hpp), and counts the tiles it is done reading in semaphore 0: the
// writer pops out_cb only behind it. It loads the next unit before gathering the last channel tile
// of this one, since compute frees x_cb exactly when it pushes that tile.
#include "api/dataflow/dataflow_api.h"
#include "../trimul_in_t1/gather.hpp"

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

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto wp = TensorAccessor(wp_args, get_common_arg_val<uint32_t>(1));
    const auto wg = TensorAccessor(wg_args, get_common_arg_val<uint32_t>(2));
    const auto mk = TensorAccessor(m_args, get_common_arg_val<uint32_t>(3));
    constexpr uint32_t o_ = m_args.next_compile_time_args_offset();
    constexpr uint32_t SPLIT = get_compile_time_arg_val(o_);
    constexpr uint32_t NB = get_compile_time_arg_val(o_ + 1);
    constexpr uint32_t CTO = get_compile_time_arg_val(o_ + 2);
    constexpr uint32_t CTT = get_compile_time_arg_val(o_ + 3);
    constexpr uint32_t OUT_TILES = get_compile_time_arg_val(o_ + 4);
    constexpr auto a_args = TensorAccessorArgs<o_ + 5>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t w_cb = tt::CBIndex::c_0;
    constexpr uint32_t x_cb = tt::CBIndex::c_1;
    constexpr uint32_t mscratch_cb = tt::CBIndex::c_3;
    constexpr uint32_t m_cb = tt::CBIndex::c_6;
    constexpr uint32_t S = St * 32;
    const uint32_t tb = get_tile_size(x_cb);

    cb_reserve_back(w_cb, 2 * CT2 * Kt);
    uint32_t w = get_write_ptr(w_cb);
    for (uint32_t t = 0; t < CT2 * Kt; ++t, w += tb) noc_async_read_page(t, wp, w);
    for (uint32_t t = 0; t < CT2 * Kt; ++t, w += tb) noc_async_read_page(t, wg, w);
    noc_async_read_barrier();
    cb_push_back(w_cb, 2 * CT2 * Kt);

    auto load = [&](uint32_t u) {
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
    };

    if constexpr (!SPLIT) {
        for (uint32_t u = u0; u < u0 + nunits; ++u) load(u);
        return;
    }
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t stage_cb = tt::CBIndex::c_25;
    constexpr uint32_t CT = CT2 / 2;
    constexpr uint32_t SS = St * St;
    const auto oa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(4));
    const auto ob = TensorAccessor(b_args, get_common_arg_val<uint32_t>(5));
    volatile tt_l1_ptr uint32_t* read_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    // out_cb is the writer's to pop: the reader follows its pages by count, never acks them.
    const uintptr_t received = (uintptr_t)get_cb_tiles_received_ptr(out_cb);
    const uint32_t out0 = get_read_ptr(out_cb);
    uint16_t seen = 0;
    uint32_t slot = 0, nq = 0;
    cb_reserve_back(stage_cb, 2 * NB);
    const uint32_t stage0 = get_write_ptr(stage_cb);
    bool dirty[2] = {false, false};

    if (nunits) load(u0);
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / SS;
        const uint32_t cube = u % SS;
        for (uint32_t q = 0; q < CT2; ++q) {
            if (q == CT2 - 1 && u + 1 < u0 + nunits) load(u + 1);
            const bool is_a = q < CT;
            const uint32_t page = ((b * CTT + CTO + (is_a ? q : q - CT)) * 32 + 16) * SS + cube;
            while ((uint16_t)((uint16_t)reg_read(received) - seen) < 32) {}
            const uint32_t src = out0 + slot * 32 * 2048;
            t1_gather_q<32, NB, 16, 32, SS, 0>(src, stage0, page, is_a, oa, ob, 0, dirty, read_done, ++nq);
            seen += 32;
            slot = slot + 1 == OUT_TILES / 32 ? 0 : slot + 1;
        }
    }
    noc_async_write_barrier();
}
