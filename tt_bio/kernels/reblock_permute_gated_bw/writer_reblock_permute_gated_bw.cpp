// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute_gated BACKWARD writer (multi-core). The gated move computes
//   a[c, i, j] = p[i, j, c] * sigmoid(g[i, j, c])
// with p and g two channel slices of the wide projection xw [1, N, N, Cw]. Its VJP, for the
// cotangent da [1, C, N, N], is
//   da'[i, j, c] = da[c, i, j]                       (the inverse move)
//   dp = da' * s,   dg = da' * p * s * (1 - s),      s = sigmoid(g)
// in the projection's own layout. The reader is reader_reblock_permute_back.cpp unchanged: it
// gathers da' tiles for group (it, jt, ct) in il-ascending order. This writer feeds the compute
// kernel the p and g tiles at the same 32 output positions and writes the two results back.
//
// The p/g reads sit here and not on the reader because the reader is the busy RISC: its gather is
// 64 local transactions a tile. This one issues 64 DRAM reads and 64 DRAM writes a group, all
// whole aligned tiles.
//
// Per group (it, jt, ct), output tile il is row i = it*32 + il:
//   xw page  (i * Nt + jt) * Ctw + off + ct
//   out page (i * Nt + jt) * Ct + ct          (dp and dg alike)
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known device-wedge cause
// in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t xw_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_off = get_common_arg_val<uint32_t>(1);  // value slice, in channel tiles
    const uint32_t g_off = get_common_arg_val<uint32_t>(2);  // gate slice, in channel tiles
    const uint32_t dp_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t dg_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t first_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);
    const uint32_t Ct = get_arg_val<uint32_t>(3);
    const uint32_t Ctw = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_p = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dp = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dg = get_compile_time_arg_val(3);
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr auto xw_args = TensorAccessorArgs<4>();
    constexpr auto dp_args = TensorAccessorArgs<xw_args.next_compile_time_args_offset()>();
    constexpr auto dg_args = TensorAccessorArgs<dp_args.next_compile_time_args_offset()>();
    const auto xw = TensorAccessor(xw_args, xw_addr);
    const auto dp = TensorAccessor(dp_args, dp_addr);
    const auto dg = TensorAccessor(dg_args, dg_addr);

    const uint32_t tile_bytes = get_tile_size(cb_p);
    const uint32_t NtCt = Nt * Ct;
    const uint32_t row_in = Nt * Ctw;
    for (uint32_t gi = 0; gi < num_groups; ++gi) {
        const uint32_t group = first_group + gi;
        const uint32_t it = group / NtCt;
        const uint32_t rem = group - it * NtCt;
        const uint32_t jt = rem / Ct;
        const uint32_t ct = rem - jt * Ct;

        cb_reserve_back(cb_p, TILE_HEIGHT);
        cb_reserve_back(cb_g, TILE_HEIGHT);
        uint32_t lp = get_write_ptr(cb_p);
        uint32_t lg = get_write_ptr(cb_g);
        uint32_t pp = (it * TILE_HEIGHT * Nt + jt) * Ctw + p_off + ct;
        uint32_t pg = (it * TILE_HEIGHT * Nt + jt) * Ctw + g_off + ct;
        for (uint32_t il = 0; il < TILE_HEIGHT; ++il) {
            noc_async_read_page(pp, xw, lp);
            noc_async_read_page(pg, xw, lg);
            lp += tile_bytes;
            lg += tile_bytes;
            pp += row_in;
            pg += row_in;
        }
        noc_async_read_barrier();
        cb_push_back(cb_p, TILE_HEIGHT);
        cb_push_back(cb_g, TILE_HEIGHT);

        cb_wait_front(cb_dp, TILE_HEIGHT);
        cb_wait_front(cb_dg, TILE_HEIGHT);
        uint32_t ldp = get_read_ptr(cb_dp);
        uint32_t ldg = get_read_ptr(cb_dg);
        uint32_t op = (it * TILE_HEIGHT) * NtCt + jt * Ct + ct;
        for (uint32_t il = 0; il < TILE_HEIGHT; ++il) {
            noc_async_write_page(op, dp, ldp);
            noc_async_write_page(op, dg, ldg);
            ldp += tile_bytes;
            ldg += tile_bytes;
            op += NtCt;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_dp, TILE_HEIGHT);
        cb_pop_front(cb_dg, TILE_HEIGHT);
    }
}
