// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fp32_ln writer (tt_bio/fp32_ln.py): the KT normed tiles of each of this core's nr row tiles from r0, from c_16 to
// pages r * KT .. of the output.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nr = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    for (uint32_t p = r0 * KT; p < (r0 + nr) * KT; ++p) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(p, o, get_read_ptr(cb_out));
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
