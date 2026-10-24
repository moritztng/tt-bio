// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_bias_rms reader. W^T ([32, C] tiles, KT of them) is read once into c_0 and a tile of bf16 ones written to
// c_2; both stay. Per unit (b, ib, jb) the z tiles (b, i, jb, :) of its 32 rows i stream to c_1, two rows (2 * KT
// tiles) per push. z is square and tile-aligned (the host checks), so there is no row or column padding.
// Common runtime args: z, W^T addresses. Per-core runtime args: first unit, unit count.
#include "api/dataflow/dataflow_api.h"

#ifndef PB_DM
#define PB_DM 0   // bench knob dit_pair_bias.RMS_DM: 1 skip z reads, 2 skip bias writes
#endif

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);   // c_pair / 32
    constexpr uint32_t NT = get_compile_time_arg_val(1);   // n / 32
    constexpr auto z_args = TensorAccessorArgs<2>();
    constexpr auto w_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    const auto sz = TensorAccessor(z_args, get_common_arg_val<uint32_t>(0));
    const auto sw = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_w = 0, cb_z = 1, cb_one = 2, R = 2;
    const uint32_t ts = get_tile_size(cb_z);    // bfp8 under PB_Z8
    const uint32_t tw = get_tile_size(cb_w);    // bf16, as the ones tile

    cb_reserve_back(cb_one, 1);
    volatile tt_l1_ptr uint32_t* one = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_one));
    for (uint32_t w = 0; w < tw / 4; ++w) {
        one[w] = 0x3F803F80u;   // two bf16 1.0
    }
    cb_push_back(cb_one, 1);

    cb_reserve_back(cb_w, KT);
    uint32_t wp = get_write_ptr(cb_w);
    for (uint32_t p = 0; p < KT; ++p, wp += tw) {
        noc_async_read_page(p, sw, wp);
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, KT);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / (NT * NT);
        const uint32_t ib = (u / NT) % NT;
        const uint32_t jb = u % NT;
        uint32_t page = ((b * NT * 32 + ib * 32) * NT + jb) * KT;
        for (uint32_t il = 0; il < 32; il += R) {
            cb_reserve_back(cb_z, R * KT);
            uint32_t zp = get_write_ptr(cb_z);
            for (uint32_t r = 0; r < R; ++r, page += NT * KT) {
                for (uint32_t c = 0; c < KT; ++c, zp += ts) {
                    if constexpr (!(PB_DM & 1)) noc_async_read_page(page + c, sz, zp);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_z, R * KT);
        }
    }
}
