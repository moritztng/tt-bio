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
    const uint32_t start_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);
    // Logical length of the permuted axis; see the ungated reader. The padding rows are read from
    // a real page so the group stays a fixed 32 pushes, and the writer zeroes them.
    const uint32_t D1 = get_arg_val<uint32_t>(3);
    const uint32_t Ct = get_arg_val<uint32_t>(4);   // channel tiles of ONE slice
    const uint32_t Ctw = get_arg_val<uint32_t>(5);  // channel tiles of the wide input

    constexpr uint32_t cb_p = 0;   // c_0
    constexpr uint32_t cb_g = 1;   // c_1
    constexpr uint32_t TILE_HEIGHT = 32;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(src_args, src_addr);

    constexpr uint32_t onetile = 1;
    const uint32_t row_stride = Nt * Ctw;
    const uint32_t end_group = start_group + num_groups;
    for (uint32_t group = start_group; group < end_group; ++group) {
        const uint32_t it = group / Nt;
        const uint32_t jt = group % Nt;
        const uint32_t row_abs = (it + it_off) * TILE_HEIGHT;
        const uint32_t rows_valid = (row_abs + TILE_HEIGHT <= D1) ? TILE_HEIGHT
                                                                  : (D1 - row_abs);
        const uint32_t row_base = it * TILE_HEIGHT;
        const uint32_t first_page = (row_base * Nt + jt) * Ctw;
        const uint32_t pad_page = jt * Ctw;  // row 0 of this tile column; always valid

        for (uint32_t ct = 0; ct < Ct; ++ct) {
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
    }
}
