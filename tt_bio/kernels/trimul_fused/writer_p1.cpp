// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul_fused writer, phase 1 standalone (bc4-bet-fusedbuild build step 1): each tilized channel tile of a half
// cube goes to DRAM, a and b as channel-major [C, S, S], only the unit's real 1 KB (faces 2h, 2h + 1).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr bool NOWRITE = get_compile_time_arg_val(1) & 16;   // ABL bit 16 (bench only): tiles dropped
    constexpr auto a_args = TensorAccessorArgs<2>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto A = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0));
    const auto B = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_t = 6;
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t h = u % 2, cube = u / 2, rt = cube / St, st = cube % St;
        for (uint32_t q = 0; q < 4; ++q) {
            for (uint32_t ab = 0; ab < 2; ++ab) {
                for (uint32_t c = 0; c < 32; ++c) {
                    cb_wait_front(cb_t, 1);
                    const uint32_t page = ((q * 32 + c) * St + rt) * St + st;
                    const uint64_t dst = ab == 0 ? A.get_noc_addr(page, h * 1024) : B.get_noc_addr(page, h * 1024);
                    if constexpr (!NOWRITE) {
                        noc_async_write(get_read_ptr(cb_t) + h * 1024, dst, 1024);
                        noc_async_writes_flushed();
                    }
                    cb_pop_front(cb_t, 1);
                }
            }
        }
    }
    noc_async_write_barrier();
}
