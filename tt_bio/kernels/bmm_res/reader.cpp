// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// bmm_res reader (tt_bio/bmm_res.py): out[c] = a[c] @ b[c] with b[c] resident in L1 for the whole channel.
// Per channel, b[c]'s KT x NT tiles go into c_1 in one reservation (c_1 holds one or two channels, so with two the
// next b[c] loads while compute still works through the current one); then a[c]'s MT row tiles stream through c_0,
// KT pages each. Row tile r of channel c is pages [(c*MT + r)*KT, +KT) of a, so every byte of a, b and out moves
// once and no core reads another core's channel.
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
    constexpr uint32_t cb_a = 0, cb_b = 1;
    constexpr auto a_args = TensorAccessorArgs<3>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto sa = TensorAccessor(a_args, a_addr);
    const auto sb = TensorAccessor(b_args, b_addr);
    const uint32_t ta = get_tile_size(cb_a), tb = get_tile_size(cb_b);

    for (uint32_t c = first; c < first + count; ++c) {
        cb_reserve_back(cb_b, KT * NT);
        uint32_t w = get_write_ptr(cb_b);
        for (uint32_t p = c * KT * NT; p < (c + 1) * KT * NT; ++p, w += tb) {
            noc_async_read_page(p, sb, w);
        }
        noc_async_read_barrier();
        cb_push_back(cb_b, KT * NT);
        for (uint32_t r = c * MT; r < (c + 1) * MT; ++r) {
            cb_reserve_back(cb_a, KT);
            w = get_write_ptr(cb_a);
            for (uint32_t k = 0; k < KT; ++k, w += ta) {
                noc_async_read_page(r * KT + k, sa, w);
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, KT);
        }
    }
}
