// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Resident weights read once and multicast to the grid rectangle, for readers whose every core needs every weight
// tile (trimul T1, the T2 tails). One core (the `sender`) reads the CB regions from DRAM as before and calls
// `send`; every other core skips the reads and calls `receive`. Receivers count themselves ready on the sender's
// semaphore 0; the sender then writes the regions and sets semaphore 1 on every core, ordered behind the data on
// the same path (pair_swiglu_ws's WS_MCAST handshake). Common args MC.. : x0, y0, x1, y1 (NOC coordinates of the
// rectangle), ncores, sender (x << 16 | y). The same bytes land at the same CB addresses, so results are unchanged.
#pragma once
#include "api/dataflow/dataflow_api.h"

namespace weight_mcast {

// n regions, addr[i] .. addr[i] + bytes[i] (bytes[i] > 0) of the sender's L1, the same addresses on every core.
template <uint32_t MC>
inline void send(const uint32_t* addr, const uint32_t* bytes, uint32_t n) {
    const uint32_t x0 = get_common_arg_val<uint32_t>(MC), y0 = get_common_arg_val<uint32_t>(MC + 1);
    const uint32_t x1 = get_common_arg_val<uint32_t>(MC + 2), y1 = get_common_arg_val<uint32_t>(MC + 3);
    const uint32_t ncores = get_common_arg_val<uint32_t>(MC + 4);
    const uint64_t m = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, 0)
                                      : get_noc_multicast_addr(x1, y1, x0, y0, 0);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    noc_semaphore_wait(ready, ncores - 1);
    noc_semaphore_set(ready, 0);
    for (uint32_t i = 0; i < n; ++i) noc_async_write_multicast(addr[i], m | addr[i], bytes[i], ncores - 1, i + 1 == n);
    *valid = 1;
    noc_semaphore_set_multicast(get_semaphore(1), m | get_semaphore(1), ncores - 1);
    noc_async_write_barrier();
}

template <uint32_t MC>
inline void receive() {
    const uint32_t sender_xy = get_common_arg_val<uint32_t>(MC + 5);
    noc_semaphore_inc(get_noc_addr(sender_xy >> 16, sender_xy & 0xffff, get_semaphore(0)), 1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 1);
}

}  // namespace weight_mcast
