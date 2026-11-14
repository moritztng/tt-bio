// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A core's share of a row-tile work list. The host lays a sorted list of n row tiles out as a uint32 ROW_MAJOR
// [ncores, P] tensor, row c holding entries c, c + ncores, ... (tt_bio/pair_swiglu_ws.py::row_list), so every core
// reads one page into L1 and walks `row_list_count` of them. Dataflow kernels only; a compute kernel repeats the
// one-line count.
#pragma once

inline uint32_t row_list_count(uint32_t me, uint32_t n, uint32_t ncores) {
    return me < n ? (n - 1 - me) / ncores + 1 : 0;
}

// Reads core `me`'s page of the list at `addr` into `l1`; the accessor's compile-time args start at ARGS_OFF.
template <uint32_t ARGS_OFF, uint32_t BYTES>
inline volatile tt_l1_ptr uint32_t* row_list_read(uint32_t me, uint32_t addr, uint32_t l1) {
    constexpr auto l_args = TensorAccessorArgs<ARGS_OFF>();
    const auto sl = TensorAccessor(l_args, addr, BYTES);
    noc_async_read(sl.get_noc_addr(me), l1, BYTES);
    noc_async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
}
