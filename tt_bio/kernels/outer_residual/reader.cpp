// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// outer_residual reader. The program walks z [B, n, n, C] in (b, jt, i, ct) order: work item w is z's tile
// (b, i, jt, ct), so a core's range is one or two (b, jt) runs of N * KT tiles. Each run's KT pi tiles (b, jt, ct)
// are read once into c_1 (two runs deep) before the first granule that needs them. Per work item the reader fetches
// the fp32 z tile into c_0 and only row 0 of the pj4 tile (b, i, ct), the 32 values the row broadcast reads, into
// c_2: the first 64 B of faces 0 and 1. Rows 1..31 of that slot are stale and never read. Common runtime args: z32,
// pi, pj4 addresses. Per-core: first work item, count.
#include "api/dataflow/dataflow_api.h"

#ifndef OR_DM
#define OR_DM 0   // bench knob outer_residual.DM: 1 skip z reads, 2 skip pj4 reads, 4 skip writes (writer)
#endif

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr uint32_t GRAN = get_compile_time_arg_val(3);
    constexpr auto z_args = TensorAccessorArgs<4>();
    constexpr auto i_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto j_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    const auto sz = TensorAccessor(z_args, get_common_arg_val<uint32_t>(0));
    const auto si = TensorAccessor(i_args, get_common_arg_val<uint32_t>(1));
    const auto sj = TensorAccessor(j_args, get_common_arg_val<uint32_t>(2));
    const uint32_t p0 = get_arg_val<uint32_t>(0);
    const uint32_t np = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_z = 0, cb_i = 1, cb_j = 2;
    constexpr uint32_t RUN = N * KT;   // work items of one (b, jt)
    constexpr uint32_t FACE = 512, ROW0 = 64;   // a bf16 face; faces 0 and 1 begin with row 0's two halves
    const uint32_t zs = get_tile_size(cb_z), bs = get_tile_size(cb_i);
    uint32_t have = 0xFFFFFFFF;
    for (uint32_t p = p0; p < p0 + np; p += GRAN) {
        const uint32_t n = p0 + np - p < GRAN ? p0 + np - p : GRAN;
        // pi of every run this granule enters, before the granule is reserved: compute pops a run only while it
        // works through granules already pushed, so this wait always clears
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t bj = (p + k) / RUN;
            if (bj != have) {
                cb_reserve_back(cb_i, KT);
                uint32_t wi = get_write_ptr(cb_i);
                for (uint32_t ct = 0; ct < KT; ++ct, wi += bs) {
                    noc_async_read_page(bj * KT + ct, si, wi);
                }
                noc_async_read_barrier();
                cb_push_back(cb_i, KT);
                have = bj;
            }
        }
        cb_reserve_back(cb_z, GRAN);
        cb_reserve_back(cb_j, GRAN);
        uint32_t wz = get_write_ptr(cb_z), wj = get_write_ptr(cb_j);
        for (uint32_t k = 0; k < n; ++k, wz += zs, wj += bs) {
            const uint32_t w = p + k;
            const uint32_t ct = w % KT;
            const uint32_t i = (w / KT) % N;
            const uint32_t bj = w / RUN;
            const uint32_t bi = (bj / NT) * N + i;   // b * N + i
            if constexpr (!(OR_DM & 1)) noc_async_read_page((bi * NT + bj % NT) * KT + ct, sz, wz);
            if constexpr (!(OR_DM & 2)) {
                noc_async_read(sj.get_noc_addr(bi * KT + ct, 0), wj, ROW0);
                noc_async_read(sj.get_noc_addr(bi * KT + ct, FACE), wj + FACE, ROW0);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_z, GRAN);
        cb_push_back(cb_j, GRAN);
    }
}
