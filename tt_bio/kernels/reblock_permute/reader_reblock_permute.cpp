// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute reader (multi-core). Single input x [1, N, N, 32] bf16 TILE.
//
// Each core owns a contiguous range of OUTPUT tile-groups
// [start_group, start_group + num_groups). A group g -> (it = g/Nt, jt = g%Nt)
// owns the 32 input tiles { (it*32 + il, jt) : il in [0,32) } whose flat DRAM
// page index is  page = (it*32 + il) * Nt + jt. The reader streams those 32 x
// tiles into CB c_0 in il-ascending order, for each owned group.
//
// Per-core CB accounting: num_groups * 32 pushes to c_0 (matched by compute).
// A core with num_groups == 0 pushes nothing and exits cleanly.
//
// This is the trimul_fused reader with the second (g) operand removed.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known
// device-wedge cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_group = get_arg_val<uint32_t>(1);
    const uint32_t num_groups = get_arg_val<uint32_t>(2);
    const uint32_t Nt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in = 0;  // c_0
    constexpr uint32_t TILE_HEIGHT = 32;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(src_args, src_addr);

    constexpr uint32_t onetile = 1;
    const uint32_t end_group = start_group + num_groups;
    for (uint32_t group = start_group; group < end_group; ++group) {
        const uint32_t it = group / Nt;
        const uint32_t jt = group % Nt;
        const uint32_t row_base = it * TILE_HEIGHT;

        for (uint32_t il = 0; il < TILE_HEIGHT; ++il) {
            const uint32_t page = (row_base + il) * Nt + jt;

            cb_reserve_back(cb_id_in, onetile);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_in);
            noc_async_read_page(page, s, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in, onetile);
        }
    }
}
