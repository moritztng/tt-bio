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
    // src_addr is the ONLY value that changes between calls at a fixed (N, buffer type, grid), so
    // it lives in the common runtime args: everything else is a pure function of the shape and the
    // work split, which lets the host cache the whole ProgramDescriptor and rewrite two scalars.
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);
    // D1 is the LOGICAL length of the permuted axis. The fold runs this op at 298, not at a
    // multiple of 32, so the last row-group is ragged: rows [288, 298) are real and [298, 320) are
    // tile padding. Reading a real page for the padding rows keeps the group a fixed 32 pushes --
    // the CB accounting, the compute kernel and the writer's 32-tile L1 window all depend on that --
    // and the writer overwrites those rows with zeros, so the value read is never used.
    //
    // The valid and the padding rows are two separate loops rather than one loop with a per-row
    // test: the same-shaped conditional in the writer's gather loop measured 10.7 us on a 97 us op,
    // and the page index is an induction variable (+Nt per row) once the test is gone.
    const uint32_t D1 = get_arg_val<uint32_t>(3);

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
        const uint32_t rows_valid = (row_base + TILE_HEIGHT <= D1) ? TILE_HEIGHT
                                                                   : (D1 - row_base);

        uint32_t page = row_base * Nt + jt;
        for (uint32_t il = 0; il < rows_valid; ++il) {
            cb_reserve_back(cb_id_in, onetile);
            noc_async_read_page(page, s, get_write_ptr(cb_id_in));
            noc_async_read_barrier();
            cb_push_back(cb_id_in, onetile);
            page += Nt;
        }
        // Tile padding: keep the group at a fixed 32 pushes by re-reading row 0 of this tile
        // column, which always exists. The writer zeroes these rows, so the value is never used.
        for (uint32_t il = rows_valid; il < TILE_HEIGHT; ++il) {
            cb_reserve_back(cb_id_in, onetile);
            noc_async_read_page(jt, s, get_write_ptr(cb_id_in));
            noc_async_read_barrier();
            cb_push_back(cb_id_in, onetile);
        }
    }
}
