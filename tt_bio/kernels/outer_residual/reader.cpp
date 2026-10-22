// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// outer_residual reader. Output page p of z [B, n, n, C] (tile (b, i, jt, ct)) needs three tiles: z32's page p, the
// pi tile (b, jt, ct) of pi [B, n, C] and the pj4 tile (b, i, ct) of pj4 [B, n, 1, C], whose row 0 is pj[b, i]. They go
// to c_0, c_1 and c_2, GRAN pages per push. Common runtime args: z32, pi, pj4 addresses. Per-core: first page, count.
#include "api/dataflow/dataflow_api.h"

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
    const uint32_t zs = get_tile_size(cb_z), bs = get_tile_size(cb_i);
    for (uint32_t p = p0; p < p0 + np; p += GRAN) {
        const uint32_t n = p0 + np - p < GRAN ? p0 + np - p : GRAN;
        cb_reserve_back(cb_z, GRAN);
        cb_reserve_back(cb_i, GRAN);
        cb_reserve_back(cb_j, GRAN);
        uint32_t wz = get_write_ptr(cb_z), wi = get_write_ptr(cb_i), wj = get_write_ptr(cb_j);
        for (uint32_t k = 0; k < n; ++k, wz += zs, wi += bs, wj += bs) {
            const uint32_t q = p + k;
            const uint32_t ct = q % KT;
            const uint32_t jt = (q / KT) % NT;
            const uint32_t bi = q / (KT * NT);   // b * N + i
            const uint32_t b = bi / N;
            noc_async_read_page(q, sz, wz);
            noc_async_read_page((b * NT + jt) * KT + ct, si, wi);
            noc_async_read_page(bi * KT + ct, sj, wj);
        }
        noc_async_read_barrier();
        cb_push_back(cb_z, GRAN);
        cb_push_back(cb_i, GRAN);
        cb_push_back(cb_j, GRAN);
    }
}
