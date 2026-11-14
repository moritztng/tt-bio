// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dit_pair_bias reader. W^T ([H, C] tiles, HT x KT) is read once into c_0 and stays. Per unit (b, ib, jb) the
// z tiles (b, i, jb, :) of the unit's valid rows i go to c_1 at slot il * KT (always 32 * KT slots reserved, so
// the CB never wraps mid-unit; rows past I are not read and compute skips them), and the padded mask tile
// (ib, jb) to c_2. z's j axis is valid up to J; in the last j tile the rows j >= J are tile padding holding
// whatever the op that wrote z left there, so they are zeroed in L1 and those columns come out as the mask's pad.
// Common runtime args: z, W^T, mask addresses. Per-core runtime args: first unit, unit count.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);   // c_pair / 32
    constexpr uint32_t HT = get_compile_time_arg_val(1);   // heads / 32
    constexpr uint32_t NT = get_compile_time_arg_val(2);   // n_key / 32 (z's j axis and both output axes)
    constexpr uint32_t I = get_compile_time_arg_val(3);    // valid rows of z's i axis
    constexpr uint32_t MB = get_compile_time_arg_val(4);   // 1: the mask has one page set per batch element
    constexpr uint32_t J = get_compile_time_arg_val(5);    // valid entries of z's j axis
    constexpr auto z_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto sz = TensorAccessor(z_args, get_common_arg_val<uint32_t>(0));
    const auto sw = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1));
    const auto sm = TensorAccessor(m_args, get_common_arg_val<uint32_t>(2));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_w = 0, cb_z = 1, cb_m = 2;
    const uint32_t ts = get_tile_size(cb_z);

    cb_reserve_back(cb_w, HT * KT);
    uint32_t wp = get_write_ptr(cb_w);
    for (uint32_t p = 0; p < HT * KT; ++p, wp += ts) {
        noc_async_read_page(p, sw, wp);
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, HT * KT);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / (NT * NT);
        const uint32_t ib = (u / NT) % NT;
        const uint32_t jb = u % NT;
        const uint32_t nv = I - ib * 32 < 32 ? I - ib * 32 : 32;
        cb_reserve_back(cb_z, 32 * KT);
        cb_reserve_back(cb_m, 1);
        uint32_t zp = get_write_ptr(cb_z);
        uint32_t page = ((b * I + ib * 32) * NT + jb) * KT;
        for (uint32_t il = 0; il < nv; ++il, page += NT * KT) {
            for (uint32_t c = 0; c < KT; ++c, zp += ts) {
                noc_async_read_page(page + c, sz, zp);
            }
        }
        noc_async_read_page((MB ? b * NT * NT : 0) + ib * NT + jb, sm, get_write_ptr(cb_m));
        noc_async_read_barrier();
        if constexpr (J % 32 != 0) {
            if (jb == NT - 1) {
                // row r of a tile: faces (r >> 4) * 2 and + 1, face row r & 15, 32 bytes each
                for (uint32_t t = 0; t < nv * KT; ++t) {
                    const uint32_t base = get_write_ptr(cb_z) + t * ts;
                    for (uint32_t r = J % 32; r < 32; ++r) {
                        const uint32_t off = (r >> 4) * 1024 + (r & 15) * 32;
                        volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + off);
                        volatile tt_l1_ptr uint32_t* b2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + off + 512);
                        for (uint32_t w = 0; w < 8; ++w) {
                            a[w] = 0;
                            b2[w] = 0;
                        }
                    }
                }
            }
        }
        cb_push_back(cb_z, 32 * KT);
        cb_push_back(cb_m, 1);
    }
}
