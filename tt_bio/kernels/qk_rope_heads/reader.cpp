// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qk_rope_heads reader. A unit is one 32-row tile of q, k or v ([B, L, D], KT tiles a row): unit u is tensor u / R,
// row tile u % R. q and k rows go to c_5 for the LayerNorm with their position's cos / sin tile (c_6 / c_7, the first
// column tile of the [1, 1, L, 64] tables: both halves are equal); v rows go to c_21, VC tiles at a time, and on to
// the writer untouched. Once, first: the 1/D reduce scaler (c_10), an fp32 eps tile (c_11) and both gammas (c_8 q,
// c_15 k, KT tiles, row 0 carries the vector); under QKRH_RELOAD (both would not fit L1 beside the rest, the 6B)
// the unit's gamma is read again into c_8 with every q / k row instead; under QKRH_NO_LN (SaProt: RoPE without a q / k
// LayerNorm) no gamma is read at all; under QKRH_RMS_HEADS (the RFdiffusion3 DiT: RMSNorm, no RoPE) no cos / sin. q, k and v may be one packed [B, L, 3D] qkv (RS = 3 * KT, TO = KT).
// Common runtime args: q, k, v, gamma q, gamma k, cos, sin addresses. Per core: first q / k unit, q / k unit count,
// first v unit, v unit count.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    const uint32_t qk_first = get_arg_val<uint32_t>(0);
    const uint32_t qk_count = get_arg_val<uint32_t>(1);
    const uint32_t v_first = get_arg_val<uint32_t>(2);
    const uint32_t v_count = get_arg_val<uint32_t>(3);
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t LT = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t SCALER = get_compile_time_arg_val(3);
    constexpr uint32_t EPS = get_compile_time_arg_val(4);
    constexpr uint32_t VC = get_compile_time_arg_val(5);
    constexpr uint32_t RS = get_compile_time_arg_val(6);   // tiles a row of the input: KT, or 3 * KT for a packed qkv
    constexpr uint32_t TO = get_compile_time_arg_val(7);   // column tile offset of k from q, v from k: 0, or KT packed
    constexpr uint32_t cb_x = 5, cb_cos = 6, cb_sin = 7, cb_gq = 8, cb_scaler = 10, cb_eps = 11, cb_gk = 15,
                       cb_v = 21;
    constexpr auto q_args = TensorAccessorArgs<8>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto gq_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto gk_args = TensorAccessorArgs<gq_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    const auto sq = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0));
    const auto sk = TensorAccessor(k_args, get_common_arg_val<uint32_t>(1));
    const auto sv = TensorAccessor(v_args, get_common_arg_val<uint32_t>(2));
    const auto sgq = TensorAccessor(gq_args, get_common_arg_val<uint32_t>(3));
    const auto sgk = TensorAccessor(gk_args, get_common_arg_val<uint32_t>(4));
    const auto sc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(5));
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(6));
    if (qk_count + v_count == 0) {
        return;
    }
    const uint32_t tx = get_tile_size(cb_x);

    generate_reduce_scaler(cb_scaler, SCALER);
    cb_reserve_back(cb_eps, 1);
    volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
    for (uint32_t i = 0; i < 1024; ++i) e[i] = EPS;
    cb_push_back(cb_eps, 1);
    uint32_t w;
#if !defined(QKRH_RELOAD) && !defined(QKRH_NO_LN)
    cb_reserve_back(cb_gq, KT);
    cb_reserve_back(cb_gk, KT);
    w = get_write_ptr(cb_gq);
    for (uint32_t p = 0; p < KT; ++p, w += tx) noc_async_read_page(p, sgq, w);
    w = get_write_ptr(cb_gk);
    for (uint32_t p = 0; p < KT; ++p, w += tx) noc_async_read_page(p, sgk, w);
    noc_async_read_barrier();
    cb_push_back(cb_gq, KT);
    cb_push_back(cb_gk, KT);
#endif

    for (uint32_t i = 0; i < qk_count + v_count; ++i) {
        const uint32_t u = i < qk_count ? qk_first + i : v_first + i - qk_count;
        const uint32_t t = u / R, r = u % R;
        if (t == 2) {
            for (uint32_t c0 = 0; c0 < KT; c0 += VC) {
                cb_reserve_back(cb_v, VC);
                w = get_write_ptr(cb_v);
                for (uint32_t c = c0; c < c0 + VC; ++c, w += tx) noc_async_read_page(r * RS + 2 * TO + c, sv, w);
                noc_async_read_barrier();
                cb_push_back(cb_v, VC);
            }
            continue;
        }
        [[maybe_unused]] const uint32_t lt = r % LT;
        cb_reserve_back(cb_x, KT);
#ifndef QKRH_RMS_HEADS
        cb_reserve_back(cb_cos, 1);
        cb_reserve_back(cb_sin, 1);
#endif
        w = get_write_ptr(cb_x);
        if (t == 0) {
            for (uint32_t c = 0; c < KT; ++c, w += tx) noc_async_read_page(r * RS + c, sq, w);
        } else {
            for (uint32_t c = 0; c < KT; ++c, w += tx) noc_async_read_page(r * RS + TO + c, sk, w);
        }
#ifdef QKRH_RELOAD
        cb_reserve_back(cb_gq, KT);
        uint32_t wg = get_write_ptr(cb_gq);
        if (t == 0) {
            for (uint32_t p = 0; p < KT; ++p, wg += tx) noc_async_read_page(p, sgq, wg);
        } else {
            for (uint32_t p = 0; p < KT; ++p, wg += tx) noc_async_read_page(p, sgk, wg);
        }
#endif
#ifndef QKRH_RMS_HEADS
        noc_async_read_page(lt * 2, sc, get_write_ptr(cb_cos));
        noc_async_read_page(lt * 2, ss, get_write_ptr(cb_sin));
#endif
        noc_async_read_barrier();
#ifdef QKRH_RELOAD
        cb_push_back(cb_gq, KT);
#endif
        cb_push_back(cb_x, KT);
#ifndef QKRH_RMS_HEADS
        cb_push_back(cb_cos, 1);
        cb_push_back(cb_sin, 1);
#endif
    }
}
