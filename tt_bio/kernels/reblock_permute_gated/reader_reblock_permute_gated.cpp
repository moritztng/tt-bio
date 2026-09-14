// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute GATED reader (multi-core). Same output tile-group ownership as
// reader_reblock_permute.cpp, but the input is the WIDE fused projection
// x [1, N, N, Cw] and each output channel tile is built from TWO channel slices
// of it: the value slice p at tile offset `p_off` and the gate slice g at
// `g_off`. The kernel that used to run over `chunk(x, 4)[2] * sigmoid(chunk(x, 4)[0])`
// now reads those two slices in place, so `ttnn.chunk` and both `ttnn.multiply_`
// calls disappear from the module.
//
// Page index of (row, jt, absolute channel tile a) in the wide tensor:
//   page = (row * Nt + jt) * Ctw + a
// Ctw is the wide tensor's channel-tile count (4 * Ct for the trimul), so the
// only change against the ungated reader is that the row stride is Nt*Ctw and
// the channel tile carries a slice offset.
//
// Both reads of a tile pair are issued BEFORE the barrier. The ungated reader
// issues one page and waits for it, and its ~2.4 us per tile per core sits just
// under the writer's 64-transaction gather, so it is hidden. Two serialised DRAM
// round trips would not be, and the reader would become the critical path; two
// in flight behind one barrier keeps the pair at roughly one round trip.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known
// device-wedge cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // The two slice offsets join the source address in the common args: they are the only values
    // that differ between the `a` and the `b` call at a fixed shape, so keeping them here lets one
    // cached ProgramDescriptor serve both.
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_off = get_common_arg_val<uint32_t>(1);  // value slice, in channel tiles
    const uint32_t g_off = get_common_arg_val<uint32_t>(2);  // gate slice, in channel tiles
    // Row-tile offset of this block inside the full permuted axis. 0 for a whole-tensor move. The
    // source block is addressed LOCALLY -- it is its own tensor -- so this only enters the padding
    // test, which asks where the block's rows sit in the logical D1.
    const uint32_t it_off = get_common_arg_val<uint32_t>(3);
    const uint32_t first_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);
    // Logical length of the permuted axis; see the ungated reader. The padding rows are read from
    // a real page so the group stays a fixed 32 pushes, and the writer zeroes them.
    const uint32_t D1 = get_arg_val<uint32_t>(3);
    const uint32_t Ct = get_arg_val<uint32_t>(4);   // channel tiles of ONE slice
    const uint32_t Ctw = get_arg_val<uint32_t>(5);  // channel tiles of the wide input
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
    const uint32_t group_stride = get_arg_val<uint32_t>(6);
    const uint32_t group_wrap_hi = get_arg_val<uint32_t>(7);
    const uint32_t group_wrap_lo = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_p = 0;   // c_0
    constexpr uint32_t cb_g = 1;   // c_1
    constexpr uint32_t TILE_HEIGHT = 32;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(src_args, src_addr);

    constexpr uint32_t onetile = 1;
    const uint32_t row_stride = Nt * Ctw;
    const uint32_t NtCt = Nt * Ct;
    uint32_t group = first_group;
    for (uint32_t gi = 0; gi < num_groups; ++gi) {
        // A group is (it, jt, ct) with ct fastest, not (it, jt) with the channel tiles looped
        // inside. At a row block the (it, jt) key leaves most of the grid idle -- 32 groups for a
        // 64-row block against 110 cores -- and this key gives 256. writer_reblock_permute_back
        // already does the same thing for the same reason.
        const uint32_t it = group / NtCt;
        const uint32_t rem = group - it * NtCt;
        const uint32_t jt = rem / Ct;
        const uint32_t ct = rem - jt * Ct;
        const uint32_t row_abs = (it + it_off) * TILE_HEIGHT;
        const uint32_t rows_valid = (row_abs + TILE_HEIGHT <= D1) ? TILE_HEIGHT
                                                                  : (D1 - row_abs);
        const uint32_t row_base = it * TILE_HEIGHT;
        const uint32_t first_page = (row_base * Nt + jt) * Ctw;
        const uint32_t pad_page = jt * Ctw;  // row 0 of this tile column; always valid

        {
            uint32_t p_page = first_page + p_off + ct;
            uint32_t g_page = first_page + g_off + ct;
            for (uint32_t il = 0; il < rows_valid; ++il) {
                cb_reserve_back(cb_p, onetile);
                cb_reserve_back(cb_g, onetile);
                noc_async_read_page(p_page, s, get_write_ptr(cb_p));
                noc_async_read_page(g_page, s, get_write_ptr(cb_g));
                noc_async_read_barrier();
                cb_push_back(cb_p, onetile);
                cb_push_back(cb_g, onetile);
                p_page += row_stride;
                g_page += row_stride;
            }
            const uint32_t p_pad = pad_page + p_off + ct;
            const uint32_t g_pad = pad_page + g_off + ct;
            for (uint32_t il = rows_valid; il < TILE_HEIGHT; ++il) {
                cb_reserve_back(cb_p, onetile);
                cb_reserve_back(cb_g, onetile);
                noc_async_read_page(p_pad, s, get_write_ptr(cb_p));
                noc_async_read_page(g_pad, s, get_write_ptr(cb_g));
                noc_async_read_barrier();
                cb_push_back(cb_p, onetile);
                cb_push_back(cb_g, onetile);
            }
        }

        group += group_stride;
        if (group == group_wrap_hi) {
            group = group_wrap_lo;
        }
    }
}
