// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// bmm_res reader (tt_bio/bmm_res.py): out[c] = a[c] @ b[c] with b[c] resident in L1 for the whole channel.
// Per channel, b[c]'s KT x NT tiles go into c_1 in one reservation (c_1 holds one or two channels, so with two the
// next b[c] loads while compute still works through the current one); then a[c]'s MT row tiles stream through c_0,
// KT pages each. Row tile r of channel c is pages [(c*MT + r)*KT, +KT) of a, so every byte of a, b and out moves
// once and no core reads another core's channel.
// TA: a is stored [K, M], so row tile r is the column of pages c*MT*KT + k*MT + r, read into c_2 for compute to
// transpose. TB: b is stored [N, K], so its tile (k, n) is page n*KT + k; c_1 still holds b in (k, n) order and the
// matmul unpacker transposes each tile.
// PF (b two channels deep): b[c+1] is read in MT slices, one after each row tile of a[c], so its load runs under
// channel c's matmuls instead of between channels.
//
// Common runtime args: a, b addresses. Per-core runtime args: first channel, channel count.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr bool TA = get_compile_time_arg_val(3);
    constexpr bool TB = get_compile_time_arg_val(4);
    constexpr bool PF = get_compile_time_arg_val(5);
    constexpr uint32_t cb_a = TA ? 2 : 0, cb_b = 1, BT = KT * NT;
    constexpr auto a_args = TensorAccessorArgs<6>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto sa = TensorAccessor(a_args, a_addr);
    const auto sb = TensorAccessor(b_args, b_addr);
    const uint32_t ta = get_tile_size(cb_a), tb = get_tile_size(cb_b);

    // b tile i = k * NT + n of channel c, written at w.
    auto read_b = [&](uint32_t c, uint32_t i, uint32_t w) {
        const uint32_t k = i / NT, n = i % NT;
        noc_async_read_page(c * BT + (TB ? n * KT + k : i), sb, w);
    };
    const uint32_t end = first + count;
    for (uint32_t c = first; c < end; ++c) {
        if (!PF || c == first) {
            cb_reserve_back(cb_b, BT);
            const uint32_t w = get_write_ptr(cb_b);
            for (uint32_t i = 0; i < BT; ++i) {
                read_b(c, i, w + i * tb);
            }
            noc_async_read_barrier();
            cb_push_back(cb_b, BT);
        }
        const bool next = PF && c + 1 < end;
        uint32_t wb = 0;
        for (uint32_t r = 0; r < MT; ++r) {
            cb_reserve_back(cb_a, KT);
            uint32_t w = get_write_ptr(cb_a);
#ifndef BRES_NO_A  // time attribution only (af2_trimul_bmm_sweep --abl)
            for (uint32_t k = 0; k < KT; ++k, w += ta) {
                noc_async_read_page(c * MT * KT + (TA ? k * MT + r : r * KT + k), sa, w);
            }
#endif
            noc_async_read_barrier();  // also lands the previous row's slice of b[c + 1]
            cb_push_back(cb_a, KT);
            if (next) {
                if (r == 0) {
                    cb_reserve_back(cb_b, BT);  // free once compute is done with b[c - 1]
                    wb = get_write_ptr(cb_b);
                }
                for (uint32_t i = r * BT / MT; i < (r + 1) * BT / MT; ++i) {
                    read_b(c + 1, i, wb + i * tb);
                }
            }
        }
        if (next) {
            noc_async_read_barrier();
            cb_push_back(cb_b, BT);
        }
    }
}
