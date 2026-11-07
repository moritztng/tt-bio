// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fp32_transition writer (tt_bio/fp32_transition.py). The G cores of a row tile each hold one K run of fc3; core j owns
// output tiles j * KT / G ... For every row tile of this core: the KT fp32 partials from c_18 go, in order m, into slot
// j * MG + m % MG of the owner's c_19 (CBs sit at the same L1 address on every core), and the owner's semaphore counts
// each peer once all of its partials have landed. Once G - 1 peers have, c_19 goes to compute, whose fixed-order sum
// leaves through c_16 to pages r * KT + j * MG .. of the output. Args: r0, nr, j, then the G owners' NoC x / y.
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t KT = get_compile_time_arg_val(0);
constexpr uint32_t G = get_compile_time_arg_val(1);
constexpr uint32_t MG = KT / G;

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<2>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nr = get_arg_val<uint32_t>(1);
    const uint32_t j = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_part = tt::CBIndex::c_18, cb_parts = tt::CBIndex::c_19, cb_out = tt::CBIndex::c_16;
    const uint32_t ftb = get_tile_size(cb_part);
    const uint32_t sem_addr = get_semaphore(0);
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    // G > 1 serves one row tile, so c_19 is reserved once before any peer can write into it.
    for (uint32_t r = r0; r < r0 + nr; ++r) {
        cb_reserve_back(cb_parts, KT);
        const uint32_t slots = get_write_ptr(cb_parts);
        for (uint32_t m = 0; m < KT; ++m) {
            const uint32_t own = m / MG;
            cb_wait_front(cb_part, 1);
            noc_async_write(get_read_ptr(cb_part),
                            get_noc_addr(get_arg_val<uint32_t>(3 + 2 * own), get_arg_val<uint32_t>(4 + 2 * own),
                                         slots + (j * MG + m % MG) * ftb), ftb);
            noc_async_writes_flushed();
            cb_pop_front(cb_part, 1);
        }
        if constexpr (G > 1) {
            noc_async_write_barrier();
            for (uint32_t i = 0; i < G; ++i)
                if (i != j)
                    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3 + 2 * i), get_arg_val<uint32_t>(4 + 2 * i),
                                                   sem_addr), 1);
            noc_semaphore_wait_min(sem, G - 1);
            // nothing reaches the semaphore after this and the next launch starts once every core has finished
            noc_semaphore_set(sem, 0);
        } else {
            noc_async_write_barrier();
        }
        cb_push_back(cb_parts, KT);
        for (uint32_t p = r * KT + j * MG; p < r * KT + (j + 1) * MG; ++p) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(p, o, get_read_ptr(cb_out));
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
