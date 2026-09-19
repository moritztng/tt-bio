// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute reader (multi-core). Single input x [1, N, N, C] bf16 TILE,
// C a multiple of 32, Ct = C/32 channel tiles.
//
// Each core owns a contiguous range of OUTPUT tile-groups
// [first_group, first_group + num_groups). A group g -> (it = g/Nt, jt = g%Nt)
// owns, for each channel tile ct, the 32 input tiles
// { (it*32 + il, jt, ct) : il in [0,32) } whose flat page index is
//   page = ((it*32 + il) * Nt + jt) * Ct + ct
// so the page stride along the permuted axis is Nt*Ct. The reader streams those
// 32 tiles into CB c_0 in il-ascending order, CT_STREAM channel tiles at a time:
// CT_STREAM consecutive pages, then the next row. CT_STREAM = 1 is one channel
// tile at a time, the original order.
//
// Per-core CB accounting: num_groups * Ct * 32 pushes to c_0 (matched by compute).
// A core with num_groups == 0 pushes nothing and exits cleanly.
//
// The channel count is NOT fixed at 32: the trunk's chunk width is
// `_trimul_chunk_size`, which doubles while the chunk still fits the L1 budget
// scaled by the compute grid, so a 13x10 grid folds 298 aa with C = 64 where an
// 11x10 grid folds it with C = 32.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known
// device-wedge cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // src_addr is the ONLY value that changes between calls at a fixed (N, C, buffer type, grid), so
    // it lives in the common runtime args: everything else is a pure function of the shape and the
    // work split, which lets the host cache the whole ProgramDescriptor and rewrite two scalars.
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);
    // D1 is the LOGICAL length of the permuted axis. The fold runs this op at 298, not at a
    // multiple of 32, so the last row-group is ragged: rows [288, 298) are real and [298, 320) are
    // tile padding. Reading a real page for the padding rows keeps the group a fixed 32 pushes --
    // the CB accounting, the compute kernel and the writer's 32-tile L1 window all depend on that --
    // and the writer overwrites those rows with zeros, so the value read is never used.
    //
    // The valid rows and the padding rows are two separate loops rather than one loop with a
    // per-row test: the same-shaped conditional in the writer's gather loop measured 10.7 us on a
    // 97 us op, and the page index is an induction variable (+Nt*Ct per row) once the test is gone.
    const uint32_t D1 = get_arg_val<uint32_t>(3);
    const uint32_t Ct = get_arg_val<uint32_t>(4);
    // The group walk is (first, stride, wrap), not (start, +1), and that is a bandwidth decision.
    // Interleaved DRAM puts page p in bank p % 8 and every page of group g is congruent to g, so
    // the cores running their i-th group together land on banks {first_k + i*stride}. A plain
    // contiguous block makes first_k = k*w, which covers 8/gcd(w, 8) of them: two of eight at
    // w = 4, measured 1.71x slower per wave than w = 5 at the same traffic. Two ways out, and the
    // host picks between them per split (see `tt_bio/reblock_permute.py`):
    //   stride = num_cores   concurrent groups become consecutive indices;
    //   wrap                 the block is kept and its START is rotated by the core's own phase,
    //                        which spreads the banks the same way without scattering the pages.
    // A core walks `num_groups` steps of `group_stride` from `first_group`, folding back to
    // `group_wrap_lo` whenever it reaches `group_wrap_hi`. Blocked order is stride 1 and a wrap
    // that never fires, so all three modes are this one loop.
    const uint32_t group_stride = get_arg_val<uint32_t>(5);
    const uint32_t group_wrap_hi = get_arg_val<uint32_t>(6);
    const uint32_t group_wrap_lo = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in = 0;  // c_0
    constexpr uint32_t TILE_HEIGHT = 32;

    // How many channel tiles are read back to back before the row index advances. Interleaved DRAM
    // puts page p in bank p % 8 and the inner page stride along the permuted axis is Nt*Ct, which is
    // 0 mod 8 whenever Ct is, so a pure il walk issues all 32 of its reads to ONE bank at an
    // outstanding depth of 1. Reading CT_STREAM consecutive pages first covers CT_STREAM banks; the
    // pages are the same pages in a different order, so the values are unchanged. CT_STREAM = 1 is
    // the original il-inner walk. The writer holds 32*CT_STREAM tiles instead of 32 in exchange,
    // which is what bounds CT_STREAM -- see `_ct_stream` in tt_bio/reblock_permute.py.
    constexpr uint32_t CT_STREAM = get_compile_time_arg_val(0);

    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto s = TensorAccessor(src_args, src_addr);

    constexpr uint32_t onetile = 1;
    const uint32_t row_stride = Nt * Ct;
    uint32_t group = first_group;
    for (uint32_t gi = 0; gi < num_groups; ++gi) {
        const uint32_t it = group / Nt;
        const uint32_t jt = group % Nt;
        const uint32_t row_base = it * TILE_HEIGHT;
        const uint32_t rows_valid = (row_base + TILE_HEIGHT <= D1) ? TILE_HEIGHT
                                                                  : (D1 - row_base);
        const uint32_t first_page = (row_base * Nt + jt) * Ct;
        const uint32_t pad_page = jt * Ct;  // row 0 of this tile column; always valid

        for (uint32_t ct0 = 0; ct0 < Ct; ct0 += CT_STREAM) {
            uint32_t page = first_page + ct0;
            for (uint32_t il = 0; il < rows_valid; ++il) {
                for (uint32_t k = 0; k < CT_STREAM; ++k) {
                    cb_reserve_back(cb_id_in, onetile);
                    noc_async_read_page(page + k, s, get_write_ptr(cb_id_in));
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in, onetile);
                }
                page += row_stride;
            }
            // Tile padding: keep the block at a fixed 32*CT_STREAM pushes. The writer zeroes these
            // rows, so the value read is never used.
            const uint32_t pad = pad_page + ct0;
            for (uint32_t il = rows_valid; il < TILE_HEIGHT; ++il) {
                for (uint32_t k = 0; k < CT_STREAM; ++k) {
                    cb_reserve_back(cb_id_in, onetile);
                    noc_async_read_page(pad + k, s, get_write_ptr(cb_id_in));
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in, onetile);
                }
            }
        }

        group += group_stride;
        if (group == group_wrap_hi) {
            group = group_wrap_lo;
        }
    }
}
