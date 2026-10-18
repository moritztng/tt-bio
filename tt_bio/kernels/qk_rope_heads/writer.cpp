// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qk_rope_heads writer. The head split is this kernel's write addresses: column tile c of row tile r = (b, lt) of a
// [B, L, D] input is tile s = c % 2 of head h = c / 2 at row tile lt, i.e. page ((b * H + h) * LT + lt) * 2 + s of
// the [B, H, L, 64] output. q and k come from the compute kernel two tiles (one head) at a time in c_16, v straight
// from the reader, VC tiles at a time, in c_21.
// Common runtime args: q, k, v output addresses. Per core: as the reader.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t qk_first = get_arg_val<uint32_t>(0);
    const uint32_t qk_count = get_arg_val<uint32_t>(1);
    const uint32_t v_first = get_arg_val<uint32_t>(2);
    const uint32_t v_count = get_arg_val<uint32_t>(3);
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t LT = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t VC = get_compile_time_arg_val(3);
    constexpr uint32_t H = KT / 2;
    constexpr uint32_t cb_out = 16, cb_v = 21;
    constexpr auto q_args = TensorAccessorArgs<4>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto sq = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0));
    const auto sk = TensorAccessor(k_args, get_common_arg_val<uint32_t>(1));
    const auto sv = TensorAccessor(v_args, get_common_arg_val<uint32_t>(2));
    const uint32_t to = get_tile_size(cb_out);
    for (uint32_t i = 0; i < qk_count + v_count; ++i) {
        const uint32_t u = i < qk_count ? qk_first + i : v_first + i - qk_count;
        const uint32_t t = u / R, r = u % R, b = r / LT, lt = r % LT;
        const uint32_t base = b * H * LT * 2 + lt * 2;   // page of head 0, tile 0; head h adds h * LT * 2
        if (t == 2) {
            for (uint32_t c0 = 0; c0 < KT; c0 += VC) {
                cb_wait_front(cb_v, VC);
                uint32_t rd = get_read_ptr(cb_v);
                for (uint32_t c = c0; c < c0 + VC; ++c, rd += to) {
                    noc_async_write_page(base + (c / 2) * LT * 2 + c % 2, sv, rd);
                }
                noc_async_writes_flushed();   // the CB slot may be reused once the data left L1
                cb_pop_front(cb_v, VC);
            }
            continue;
        }
        for (uint32_t h = 0; h < H; ++h) {
            cb_wait_front(cb_out, 2);
            uint32_t rd = get_read_ptr(cb_out);
            const uint32_t p = base + h * LT * 2;
            if (t == 0) {
                noc_async_write_page(p, sq, rd);
                noc_async_write_page(p + 1, sq, rd + to);
            } else {
                noc_async_write_page(p, sk, rd);
                noc_async_write_page(p + 1, sk, rd + to);
            }
            noc_async_writes_flushed();   // the CB slot may be reused once the data left L1
            cb_pop_front(cb_out, 2);
        }
    }
    noc_async_write_barrier();
}
