// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RFD3 dense attention-score reader: streams the bf16 scores and the bf16 attention bias, one
// tile page of each, so `compute_fused_scores.cpp` can widen both and write
// `scores * scale + bias` in fp32 without either fp32 operand ever existing in DRAM.
//
// Together with compute_fused_scores.cpp and writer_dense_scores.cpp this replaces three ops on
// the DiT's dense branch:
//     typecast(bias, fp32) -> typecast(scores, fp32)
//     -> add(scores * scale, bias)
// At the page fixture's shape, [1, 16, 685, 704], that chain reads 30.9 and writes 92.7 MB per
// call; this pass reads 30.9 and writes 30.9.
//
// The two tensors and the output have identical tile-page layouts, so page `p` of one is page `p`
// of the others and no band geometry has to be carried anywhere: a core owns a flat page range.
// That is also why the work splits evenly -- 7744 pages over 130 cores is 59 or 60 each, where the
// sparse kernel's (head, tile-row) bands would have been 352 over 130, a 1.5x imbalance.
//
// Reads are issued WINDOW pages deep and barriered once, because a barrier per page would
// serialise the DRAM latency 60 times per core. The ring slot is tracked here rather than read
// from the CB, since a window can straddle the wrap once the count is not a multiple of SLOTS --
// the same reason writer_fused_scores.cpp tracks its own counter.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known device-wedge
// cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t scores_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_common_arg_val<uint32_t>(1);

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_scores = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(2);   // both CBs' depth, power of two
    constexpr uint32_t WINDOW = get_compile_time_arg_val(3);  // pages per barrier, <= SLOTS
    constexpr auto scores_args = TensorAccessorArgs<4>();
    constexpr auto bias_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();

    constexpr uint32_t TILE_BYTES = 32 * 32 * 2;

    const auto s_scores = TensorAccessor(scores_args, scores_addr);
    const auto s_bias = TensorAccessor(bias_args, bias_addr);

    // Captured before any push, so ring page `s` is base + s * TILE_BYTES.
    const uint32_t scores_base = get_write_ptr(cb_scores);
    const uint32_t bias_base = get_write_ptr(cb_bias);

    uint32_t pushed = 0;
    uint32_t page = start_page;
    uint32_t left = num_pages;
    while (left) {
        uint32_t n = (left < WINDOW) ? left : WINDOW;
        cb_reserve_back(cb_scores, n);
        cb_reserve_back(cb_bias, n);
        for (uint32_t w = 0; w < n; ++w) {
            const uint32_t slot = (pushed + w) & (SLOTS - 1);
            noc_async_read(s_scores.get_noc_addr(page + w), scores_base + slot * TILE_BYTES,
                           TILE_BYTES);
            noc_async_read(s_bias.get_noc_addr(page + w), bias_base + slot * TILE_BYTES,
                           TILE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(cb_scores, n);
        cb_push_back(cb_bias, n);
        pushed += n;
        page += n;
        left -= n;
    }
}
