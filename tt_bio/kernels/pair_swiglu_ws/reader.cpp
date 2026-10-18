// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_swiglu_ws reader. The weights are read ONCE into L1 and stay there (c_1 = [w1 | w2], c_2 = w3);
// then this core's row tiles of x stream through c_0, KT tiles each. A row tile is 32 rows of the
// flattened pair tensor, so row tile r is pages [r*KT, (r+1)*KT) of x and nothing crosses cores.
//
// Under WS_LN the raw x streams into c_5 instead and the LayerNorm's gamma / beta (Kt tiles each, row 0 carries
// the [K] vector), the 1/K reduce scaler and an fp32 eps tile are read once into c_8 / c_9 / c_10 / c_11.
//
// Under WS_RESID the residual's row tile (CT pages) follows each x row tile into c_20, and under WS_RESID2 the second
// residual's into c_24 (common runtime arg 8).
//
// Under WS_RELU c_1 holds w1 alone ([KT, NH] tiles) and the row-replicated biases are read once into c_21 (b1, NH
// tiles) and c_22 (b3, CT tiles). Under WS_RELU_BW b1 is read the same way and the cotangent dy's row tile (KT pages)
// follows each x row tile into c_23.
//
// Common runtime args: x, w12, w3, gamma, beta, residual, b1, b3 (WS_RELU_BW: dy) addresses. Per-core runtime args: first row tile, row tile count.
#include "api/dataflow/dataflow_api.h"
#ifdef WS_LN
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#endif

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w12_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w3_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t KT = get_compile_time_arg_val(0);   // K tiles of x (c_z / 32)
    constexpr uint32_t NH = get_compile_time_arg_val(1);   // hidden tiles of one of w1, w2
    constexpr uint32_t CT = get_compile_time_arg_val(2);   // output tiles (c_z / 32)
    constexpr uint32_t SCALER = get_compile_time_arg_val(3);  // 1/K as a packed bf16 pair (WS_LN only)
    constexpr uint32_t EPS = get_compile_time_arg_val(4);     // eps as fp32 bits (WS_LN only)
#ifdef WS_LN
    constexpr uint32_t cb_x = 5;
#else
    constexpr uint32_t cb_x = 0;
#endif
    constexpr uint32_t cb_w12 = 1, cb_w3 = 2;
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto w12_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w3_args = TensorAccessorArgs<w12_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, x_addr);
    const auto s12 = TensorAccessor(w12_args, w12_addr);
    const auto s3 = TensorAccessor(w3_args, w3_addr);
#ifdef WS_LN
    constexpr auto g_args = TensorAccessorArgs<w3_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto sg = TensorAccessor(g_args, get_common_arg_val<uint32_t>(3));
    const auto sb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(4));
#endif
#ifdef WS_RESID
#ifdef WS_LN
    constexpr auto r_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
#else
    constexpr auto r_args = TensorAccessorArgs<w3_args.next_compile_time_args_offset()>();
#endif
    const auto sr = TensorAccessor(r_args, get_common_arg_val<uint32_t>(5));
    constexpr uint32_t cb_res = 20;
    const uint32_t tr = get_tile_size(cb_res);
#ifdef WS_RESID2
    constexpr auto r2_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    const auto sr2 = TensorAccessor(r2_args, get_common_arg_val<uint32_t>(8));
    constexpr uint32_t cb_res2 = 24;
#endif
#endif
#if defined(WS_RELU) || defined(WS_RELU_BW)
    // the bias accessors follow the last tensor the reader already takes (WS_RELU is never combined with WS_RESID)
#ifdef WS_LN
    constexpr uint32_t TAIL = b_args.next_compile_time_args_offset();
#else
    constexpr uint32_t TAIL = w3_args.next_compile_time_args_offset();
#endif
    constexpr uint32_t cb_b1 = 21;
    constexpr auto b1_args = TensorAccessorArgs<TAIL>();
    constexpr auto b3_args = TensorAccessorArgs<b1_args.next_compile_time_args_offset()>();
    const auto sb1 = TensorAccessor(b1_args, get_common_arg_val<uint32_t>(6));
    const auto sb3 = TensorAccessor(b3_args, get_common_arg_val<uint32_t>(7));   // WS_RELU_BW: dy
#endif
#ifdef WS_RELU
    constexpr uint32_t W12 = NH, cb_b3 = 22;
#else
    constexpr uint32_t W12 = 2 * NH;
#endif
#ifdef WS_RELU_BW
    constexpr uint32_t cb_dy = 23;
    const uint32_t tdy = get_tile_size(cb_dy);
#endif
    if (count == 0) {
        return;
    }

    // Weights in page order: w12 is [KT, W12] tiles, w3 is [NH, CT].
    const uint32_t t12 = get_tile_size(cb_w12), t3 = get_tile_size(cb_w3), tx = get_tile_size(cb_x);
    cb_reserve_back(cb_w12, KT * W12);
    uint32_t w = get_write_ptr(cb_w12);
    for (uint32_t p = 0; p < KT * W12; ++p, w += t12) {
        noc_async_read_page(p, s12, w);
    }
    cb_reserve_back(cb_w3, NH * CT);
    w = get_write_ptr(cb_w3);
    for (uint32_t p = 0; p < NH * CT; ++p, w += t3) {
        noc_async_read_page(p, s3, w);
    }
#ifdef WS_LN
    constexpr uint32_t cb_gamma = 8, cb_beta = 9, cb_scaler = 10, cb_eps = 11;
    generate_reduce_scaler(cb_scaler, SCALER);
    cb_reserve_back(cb_eps, 1);
    volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
    for (uint32_t i = 0; i < 1024; ++i) e[i] = EPS;
    cb_push_back(cb_eps, 1);
    cb_reserve_back(cb_gamma, KT);
    cb_reserve_back(cb_beta, KT);
    w = get_write_ptr(cb_gamma);
    for (uint32_t p = 0; p < KT; ++p, w += tx) noc_async_read_page(p, sg, w);
    w = get_write_ptr(cb_beta);
    for (uint32_t p = 0; p < KT; ++p, w += tx) noc_async_read_page(p, sb, w);
#endif
#if defined(WS_RELU) || defined(WS_RELU_BW)
    const uint32_t tb = get_tile_size(cb_b1);
    cb_reserve_back(cb_b1, NH);
    w = get_write_ptr(cb_b1);
    for (uint32_t p = 0; p < NH; ++p, w += tb) noc_async_read_page(p, sb1, w);
#endif
#ifdef WS_RELU
    cb_reserve_back(cb_b3, CT);
    w = get_write_ptr(cb_b3);
    for (uint32_t p = 0; p < CT; ++p, w += tb) noc_async_read_page(p, sb3, w);
#endif
    noc_async_read_barrier();
#if defined(WS_RELU) || defined(WS_RELU_BW)
    cb_push_back(cb_b1, NH);
#endif
#ifdef WS_RELU
    cb_push_back(cb_b3, CT);
#endif
#ifdef WS_LN
    cb_push_back(cb_gamma, KT);
    cb_push_back(cb_beta, KT);
#endif
    cb_push_back(cb_w12, KT * W12);
    cb_push_back(cb_w3, NH * CT);

    for (uint32_t r = first; r < first + count; ++r) {
        cb_reserve_back(cb_x, KT);
        w = get_write_ptr(cb_x);
#ifndef WS_NO_XREAD  // time attribution only (ws_attr.py)
        for (uint32_t k = 0; k < KT; ++k, w += tx) {
            noc_async_read_page(r * KT + k, sx, w);
        }
#endif
#ifdef WS_RELU_BW
        cb_reserve_back(cb_dy, KT);
        uint32_t wd = get_write_ptr(cb_dy);
        for (uint32_t k = 0; k < KT; ++k, wd += tdy) {
            noc_async_read_page(r * KT + k, sb3, wd);
        }
#endif
#ifdef WS_RESID
        cb_reserve_back(cb_res, CT);
        uint32_t wr = get_write_ptr(cb_res);
        for (uint32_t c = 0; c < CT; ++c, wr += tr) {
            noc_async_read_page(r * CT + c, sr, wr);
        }
#ifdef WS_RESID2
        cb_reserve_back(cb_res2, CT);
        uint32_t wr2 = get_write_ptr(cb_res2);
        for (uint32_t c = 0; c < CT; ++c, wr2 += tr) {
            noc_async_read_page(r * CT + c, sr2, wr2);
        }
#endif
#endif
        noc_async_read_barrier();
        cb_push_back(cb_x, KT);
#ifdef WS_RELU_BW
        cb_push_back(cb_dy, KT);
#endif
#ifdef WS_RESID
        cb_push_back(cb_res, CT);
#ifdef WS_RESID2
        cb_push_back(cb_res2, CT);
#endif
#endif
    }
}
