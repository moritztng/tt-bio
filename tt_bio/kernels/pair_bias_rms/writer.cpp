// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_bias_rms writer. Per unit (b, ib, jb) compute hands over the 32 head tiles in head order. The heads are split
// over consecutive outputs of HO heads each, of which the first HV are real (the weight's rows past them are zero
// padding up to the head tile): head h goes to page ((b * HV + h % HO) * NT + ib) * NT + jb of output h / HO
// ([B, HV, n, n], one address per output in the common args), and a padding head is dropped.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t SW = get_compile_time_arg_val(1);
    constexpr uint32_t HO = get_compile_time_arg_val(2);
    constexpr uint32_t HV = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<4>();
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16;
    const uint32_t ts = get_tile_size(cb_out);
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t b = u / (NT * NT);
        const uint32_t ib = (u / NT) % NT;
        const uint32_t jb = u % NT;
        for (uint32_t h0 = 0; h0 < 32; h0 += SW) {
            cb_wait_front(cb_out, SW);
            uint32_t rd = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < SW; ++t, rd += ts) {
                const uint32_t h = h0 + t;
                if (h % HO < HV) {
                    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(h / HO));
                    noc_async_write_page(((b * HV + h % HO) * NT + ib) * NT + jb, so, rd);
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, SW);
        }
    }
    noc_async_write_barrier();
}
