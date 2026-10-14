// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// adaln reader (tt_bio/adaln.py). Once: the 1/K reduce scaler into c_10 and an fp32 eps tile into
// c_11. Then per row tile r of this core: Kt tiles of x into c_7, of the scale into c_3 and of the
// shift into c_4 (row tile r is pages r * Kt .. of each).
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t SCALER = get_compile_time_arg_val(1);   // packed bf16 pair, 1/K
    constexpr uint32_t EPS = get_compile_time_arg_val(2);      // fp32 bits
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto sc_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sb_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto sc = TensorAccessor(sc_args, get_common_arg_val<uint32_t>(1));
    const auto sb = TensorAccessor(sb_args, get_common_arg_val<uint32_t>(2));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nrows = get_arg_val<uint32_t>(1);

    constexpr uint32_t sc_cb = tt::CBIndex::c_3;
    constexpr uint32_t sb_cb = tt::CBIndex::c_4;
    constexpr uint32_t x_cb = tt::CBIndex::c_7;
    constexpr uint32_t scaler_cb = tt::CBIndex::c_10;
    constexpr uint32_t eps_cb = tt::CBIndex::c_11;

    generate_reduce_scaler(scaler_cb, SCALER);
    {
        cb_reserve_back(eps_cb, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(eps_cb));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(eps_cb, 1);
    }

    const uint32_t tx = get_tile_size(x_cb), tsc = get_tile_size(sc_cb), tsb = get_tile_size(sb_cb);
    for (uint32_t r = r0; r < r0 + nrows; ++r) {
        const uint32_t page = r * Kt;
        cb_reserve_back(x_cb, Kt);
        cb_reserve_back(sc_cb, Kt);
        cb_reserve_back(sb_cb, Kt);
        uint32_t dx = get_write_ptr(x_cb), dsc = get_write_ptr(sc_cb), dsb = get_write_ptr(sb_cb);
        for (uint32_t t = 0; t < Kt; ++t, dx += tx, dsc += tsc, dsb += tsb) {
            noc_async_read_page(page + t, x, dx);
            noc_async_read_page(page + t, sc, dsc);
            noc_async_read_page(page + t, sb, dsb);
        }
        noc_async_read_barrier();
        cb_push_back(x_cb, Kt);
        cb_push_back(sc_cb, Kt);
        cb_push_back(sb_cb, Kt);
    }
}
