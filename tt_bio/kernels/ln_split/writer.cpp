// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ln_split writer (tt_bio/ln_split.py). The G cores of a row tile r each hold KC of its KT column tiles. Twice (the
// row sum, then the row sum of (x - mean)^2, both already scaled by 1/K) this core's fp32 partial tile, c_12 then c_15,
// is written into slot g of every group member's c_13 / c_18 (CBs sit at the same L1 address on every core), and that
// round's semaphore on the member counts it. Once G - 1 peers have landed, the G slots go to compute, whose sum is the mean / the
// variance. Then the KC normed tiles from c_2 go to pages r * KT + k0 .. of the output.
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t G = get_compile_time_arg_val(0);
constexpr uint32_t KT = get_compile_time_arg_val(1);
constexpr uint32_t KC = get_compile_time_arg_val(2);

void exchange(uint32_t part_cb, uint32_t parts_cb, uint32_t g, volatile tt_l1_ptr uint32_t* sem, uint32_t sem_addr,
              uint32_t want) {
    const uint32_t ftb = get_tile_size(part_cb);
    cb_reserve_back(parts_cb, G);
    const uint32_t slot = get_write_ptr(parts_cb) + g * ftb;
    cb_wait_front(part_cb, 1);
    const uint32_t src = get_read_ptr(part_cb);
    for (uint32_t j = 0; j < G; ++j)
        noc_async_write(src, get_noc_addr(get_arg_val<uint32_t>(3 + 2 * j), get_arg_val<uint32_t>(4 + 2 * j), slot), ftb);
    noc_async_write_barrier();
    for (uint32_t j = 0; j < G; ++j)
        if (j != g)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3 + 2 * j), get_arg_val<uint32_t>(4 + 2 * j), sem_addr), 1);
    cb_pop_front(part_cb, 1);
    noc_semaphore_wait_min(sem, want);
    cb_push_back(parts_cb, G);
}

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<3>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t k0 = get_arg_val<uint32_t>(1);
    const uint32_t g = get_arg_val<uint32_t>(2);
    // One semaphore a round: with one, a fast peer's round-2 count could stand in for a slow peer's round-1 tile.
    const uint32_t s1 = get_semaphore(0), s2 = get_semaphore(1);
    auto* p1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s1);
    auto* p2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s2);
    exchange(tt::CBIndex::c_12, tt::CBIndex::c_13, g, p1, s1, G - 1);
    exchange(tt::CBIndex::c_15, tt::CBIndex::c_18, g, p2, s2, G - 1);
    // Nothing reaches either semaphore after round 2, and the next launch of this cached program starts after every
    // core of this one has finished, so both may be cleared here.
    noc_semaphore_set(p1, 0);
    noc_semaphore_set(p2, 0);

    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    const uint32_t tb = get_tile_size(out_cb);
    cb_wait_front(out_cb, KC);
    uint32_t src = get_read_ptr(out_cb);
    for (uint32_t t = 0; t < KC; ++t, src += tb) noc_async_write_page(r * KT + k0 + t, o, src);
    noc_async_write_barrier();
    cb_pop_front(out_cb, KC);
}
