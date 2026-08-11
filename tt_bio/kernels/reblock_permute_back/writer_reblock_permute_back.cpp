// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute_back writer (multi-core). The INVERSE channel move:
// permute(x, (0,2,3,1)) for x [1, C, N, N] -> [1, N, N, C].
//
// All the work is upstream: the reader gathers, the compute kernel transposes. This
// writer only streams finished output tiles from c_16 to DRAM, and every one of
// them is an aligned, contiguous 2 KB full-tile write, which is the whole reason
// the back direction is cheaper than the forward one. For a group (it, jt, ct) the
// compute kernel produces the 32 tiles in il-ascending order and tile il belongs at
//   page = (it*32 + il) * Nt*Ct + jt*Ct + ct.
// Distinct groups own distinct output pages, so there is nothing to synchronise.
//
// The whole group's 32 tiles are waited for, issued and drained together rather
// than one at a time: a per-tile barrier would serialise 32 independent 2 KB DRAM
// writes on the only RISC that has nothing else to do.
//
// Per-core CB accounting: num_groups * 32 pops from c_16, matched by compute.
// A core with num_groups == 0 pops nothing and exits cleanly.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known
// device-wedge cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);
    const uint32_t Ct = get_arg_val<uint32_t>(3);

    constexpr uint32_t element_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);     // c_16
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(2);   // 32
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(3);    // 32
    constexpr auto dst_args = TensorAccessorArgs<4>();

    constexpr uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * element_size;  // 2048

    const auto s = TensorAccessor(dst_args, dst_addr);

    const uint32_t NtCt = Nt * Ct;
    const uint32_t end_group = start_group + num_groups;
    for (uint32_t group = start_group; group < end_group; ++group) {
        const uint32_t it = group / NtCt;
        const uint32_t rem = group - it * NtCt;
        const uint32_t jt = rem / Ct;
        const uint32_t ct = rem - jt * Ct;

        cb_wait_front(cb_id_out, TILE_HEIGHT);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        uint32_t out_page = (it * TILE_HEIGHT) * NtCt + jt * Ct + ct;
        for (uint32_t il = 0; il < TILE_HEIGHT; ++il) {
            noc_async_write(l1_read_addr, s.get_noc_addr(out_page), tile_bytes);
            l1_read_addr += tile_bytes;
            out_page += NtCt;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, TILE_HEIGHT);
    }
}
