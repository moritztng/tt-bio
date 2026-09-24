// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// page_copy: copy whole pages (tiles) of one interleaved DRAM tensor into another, at an offset.
//
// Source tile t lands at destination page dst_off + (t / run_len) * dst_stride + t % run_len, and
// is read from src_off + (t / run_len) * src_stride + t % run_len. One run is the whole block for
// a row block of a [1, S, S, C] pair (its rows are one contiguous page range) and S runs for a
// column strip. Nothing is recomputed or reformatted: the bytes of a tile move as they are.
//
// Each core owns `n` consecutive source tiles starting at run `r`, offset `o` (the host does the
// one division). The L1 window is two halves of `depth` pages: a half is refilled only after the
// writes issued from it have left L1, so the reads of one batch overlap the writes of the last.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known device-wedge cause
// in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t src_off = get_common_arg_val<uint32_t>(2);
    const uint32_t dst_off = get_common_arg_val<uint32_t>(3);
    const uint32_t n = get_arg_val<uint32_t>(0);
    uint32_t r = get_arg_val<uint32_t>(1);
    uint32_t o = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr uint32_t run_len = get_compile_time_arg_val(2);
    constexpr uint32_t src_stride = get_compile_time_arg_val(3);
    constexpr uint32_t dst_stride = get_compile_time_arg_val(4);
    constexpr uint32_t page = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, src_addr);
    const auto dst = TensorAccessor(dst_args, dst_addr);

    if (n == 0) {
        return;
    }
    cb_reserve_back(cb, 2 * depth);
    const uint32_t l1 = get_write_ptr(cb);

    uint32_t src_page = src_off + r * src_stride + o;
    uint32_t dst_page = dst_off + r * dst_stride + o;
    uint32_t half = 0;
    for (uint32_t done = 0; done < n; done += depth) {
        const uint32_t k_n = (n - done < depth) ? (n - done) : depth;
        const uint32_t base = l1 + half * depth * page;
        noc_async_writes_flushed();
        uint32_t sp = src_page, oo = o;
        for (uint32_t k = 0; k < k_n; ++k) {
            noc_async_read(src.get_noc_addr(sp), base + k * page, page);
            if (++oo == run_len) {
                oo = 0;
                sp += src_stride - run_len + 1;
            } else {
                ++sp;
            }
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < k_n; ++k) {
            noc_async_write(base + k * page, dst.get_noc_addr(dst_page), page);
            if (++o == run_len) {
                o = 0;
                dst_page += dst_stride - run_len + 1;
            } else {
                ++dst_page;
            }
        }
        src_page = sp;
        half ^= 1;
    }
    noc_async_write_barrier();
}
