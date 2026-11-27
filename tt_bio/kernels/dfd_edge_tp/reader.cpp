// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DiffDock edge-tile reader: the group's constants once (L2 weight tiles in wi-major order, the
// nonzero G blocks, 32 one-hot tiles and a zero tile), then per edge tile its H, A and the
// broadcast sh tiles the group uses (BJ_LIST, page offsets within an edge tile's row of tiles).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t bj_list[] = {BJ_LIST};

void kernel_main() {
    constexpr uint32_t KH = get_compile_time_arg_val(0);
    constexpr uint32_t NBJ = get_compile_time_arg_val(1);
    constexpr uint32_t KA = get_compile_time_arg_val(2);
    constexpr uint32_t KB = get_compile_time_arg_val(3);   // tiles in one row of Bfull
    constexpr uint32_t NW = get_compile_time_arg_val(4);
    constexpr uint32_t NG = get_compile_time_arg_val(5);
    constexpr auto h_args = TensorAccessorArgs<6>();
    constexpr auto a_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto l2_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<l2_args.next_compile_time_args_offset()>();
    constexpr auto oh_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t n_et = get_arg_val<uint32_t>(1);
    const auto sh = TensorAccessor(h_args, get_common_arg_val<uint32_t>(0));
    const auto sa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(1));
    const auto sb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(2));
    const auto sl2 = TensorAccessor(l2_args, get_common_arg_val<uint32_t>(4));
    const auto sg = TensorAccessor(g_args, get_common_arg_val<uint32_t>(5));
    const auto soh = TensorAccessor(oh_args, get_common_arg_val<uint32_t>(3));

    constexpr uint32_t cb_h = 0, cb_a = 1, cb_b = 2, cb_l2 = 3, cb_g = 4, cb_oh = 5;
    const uint32_t tb = get_tile_size(cb_h);

    // L2 is [KH*32, NW*32]: tile (kt, wi) is page kt*NW + wi; the compute wants wi*KH + kt.
    cb_reserve_back(cb_l2, NW * KH);
    uint32_t w = get_write_ptr(cb_l2);
    for (uint32_t wi = 0; wi < NW; ++wi)
        for (uint32_t kt = 0; kt < KH; ++kt, w += tb) noc_async_read_page(kt * NW + wi, sl2, w);
    cb_reserve_back(cb_g, NG);
    w = get_write_ptr(cb_g);
    for (uint32_t i = 0; i < NG; ++i, w += tb) noc_async_read_page(i, sg, w);
    cb_reserve_back(cb_oh, 33);
    w = get_write_ptr(cb_oh);
    for (uint32_t i = 0; i < 33; ++i, w += tb) noc_async_read_page(i, soh, w);
    noc_async_read_barrier();
    cb_push_back(cb_l2, NW * KH);
    cb_push_back(cb_g, NG);
    cb_push_back(cb_oh, 33);

    for (uint32_t et = first; et < first + n_et; ++et) {
        cb_reserve_back(cb_h, KH);
        cb_reserve_back(cb_a, KA);
        cb_reserve_back(cb_b, NBJ);
        uint32_t wh = get_write_ptr(cb_h), wa = get_write_ptr(cb_a), wb = get_write_ptr(cb_b);
        for (uint32_t k = 0; k < KH; ++k, wh += tb) noc_async_read_page(et * KH + k, sh, wh);
        for (uint32_t k = 0; k < KA; ++k, wa += tb) noc_async_read_page(et * KA + k, sa, wa);
        for (uint32_t k = 0; k < NBJ; ++k, wb += tb) noc_async_read_page(et * KB + bj_list[k], sb, wb);
        noc_async_read_barrier();
        cb_push_back(cb_h, KH);
        cb_push_back(cb_a, KA);
        cb_push_back(cb_b, NBJ);
    }
}
