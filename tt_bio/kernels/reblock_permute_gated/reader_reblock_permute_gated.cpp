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

    // How many channel tiles a group covers, and therefore how many pages of each slice are read
    // back to back before the row index advances. The row stride along the permuted axis is
    // Nt*Ctw, which is 0 mod 8 at every shape the trimul runs, so a group that holds ONE channel
    // tile issues all of its reads to the two banks p_off+ct and g_off+ct and to no others. At
    // CT_STREAM > 1 the group holds CT_STREAM consecutive channel tiles of each slice, so the
    // reads of one row cover up to 2*CT_STREAM banks, and they go out behind a single barrier
    // instead of ceil/CT_STREAM of them. Same pages, same transactions, same values, different
    // order and different depth. CT_STREAM = 1 is the shipped walk, from this source.
    //
    // The group index carries the channel-tile BLOCK rather than the channel tile, so the work
    // split has Nrt*Nt*(Ct/CT_STREAM) units. What bounds CT_STREAM is the writer, which holds
    // 32*CT_STREAM tiles instead of 32; see `_ct_stream` in tt_bio/reblock_permute.py.
    constexpr uint32_t CT_STREAM = get_compile_time_arg_val(0);
    // How many ROWS are issued behind one barrier. Orthogonal to CT_STREAM and it exists to
    // separate the two mechanisms: CT_STREAM adds banks and depth together, ROW_BATCH adds depth
    // alone, because every row of a group sits on the same bank. It also costs nothing in the
    // writer -- the CB order is still il-ascending with the CT_STREAM block innermost, so the
    // group key, the work split and the writer are all untouched. ROW_BATCH = 1 is the shipped
    // walk, from this source.
    constexpr uint32_t ROW_BATCH = get_compile_time_arg_val(1);
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t BATCH_TILES = CT_STREAM * ROW_BATCH;

    constexpr auto src_args = TensorAccessorArgs<3>();
    const auto s = TensorAccessor(src_args, src_addr);

    const uint32_t row_stride = Nt * Ctw;
    const uint32_t Cs = Ct / CT_STREAM;   // channel-tile blocks per (row tile, col tile)
    const uint32_t NtCs = Nt * Cs;
    uint32_t group = first_group;
    for (uint32_t gi = 0; gi < num_groups; ++gi) {
        // A group is (row-tile, col-tile, channel-tile block) with the block fastest, not
        // (it, jt) with the channel tiles looped inside. At a row block the (it, jt) key leaves
        // most of the grid idle -- 32 groups for a 64-row block against 110 cores -- and this key
        // gives 256/CT_STREAM. writer_reblock_permute_back already does the same thing.
        const uint32_t it = group / NtCs;
        const uint32_t rem = group - it * NtCs;
        const uint32_t jt = rem / Cs;
        const uint32_t ct0 = (rem - jt * Cs) * CT_STREAM;
        const uint32_t row_abs = (it + it_off) * TILE_HEIGHT;
        const uint32_t rows_valid = (row_abs + TILE_HEIGHT <= D1) ? TILE_HEIGHT
                                                                  : (D1 - row_abs);
        const uint32_t row_base = it * TILE_HEIGHT;
        const uint32_t first_page = (row_base * Nt + jt) * Ctw;
        const uint32_t pad_page = jt * Ctw;  // row 0 of this tile column; always valid

        {
            uint32_t p_page = first_page + p_off + ct0;
            uint32_t g_page = first_page + g_off + ct0;
            uint32_t il = 0;
            for (; il + ROW_BATCH <= rows_valid; il += ROW_BATCH) {
                cb_reserve_back(cb_p, BATCH_TILES);
                cb_reserve_back(cb_g, BATCH_TILES);
                uint32_t wp = get_write_ptr(cb_p);
                uint32_t wg = get_write_ptr(cb_g);
                for (uint32_t r = 0; r < ROW_BATCH; ++r) {
                    for (uint32_t k = 0; k < CT_STREAM; ++k) {
                        noc_async_read_page(p_page + k, s, wp + k * TILE_BYTES);
                        noc_async_read_page(g_page + k, s, wg + k * TILE_BYTES);
                    }
                    wp += CT_STREAM * TILE_BYTES;
                    wg += CT_STREAM * TILE_BYTES;
                    p_page += row_stride;
                    g_page += row_stride;
                }
                noc_async_read_barrier();
                cb_push_back(cb_p, BATCH_TILES);
                cb_push_back(cb_g, BATCH_TILES);
            }
            // Rows left over when ROW_BATCH does not divide the valid row count, one at a time.
            for (; il < rows_valid; ++il) {
                cb_reserve_back(cb_p, CT_STREAM);
                cb_reserve_back(cb_g, CT_STREAM);
                const uint32_t wp = get_write_ptr(cb_p);
                const uint32_t wg = get_write_ptr(cb_g);
                for (uint32_t k = 0; k < CT_STREAM; ++k) {
                    noc_async_read_page(p_page + k, s, wp + k * TILE_BYTES);
                    noc_async_read_page(g_page + k, s, wg + k * TILE_BYTES);
                }
                noc_async_read_barrier();
                cb_push_back(cb_p, CT_STREAM);
                cb_push_back(cb_g, CT_STREAM);
                p_page += row_stride;
                g_page += row_stride;
            }
            const uint32_t p_pad = pad_page + p_off + ct0;
            const uint32_t g_pad = pad_page + g_off + ct0;
            for (il = rows_valid; il < TILE_HEIGHT; ++il) {
                cb_reserve_back(cb_p, CT_STREAM);
                cb_reserve_back(cb_g, CT_STREAM);
                const uint32_t wp = get_write_ptr(cb_p);
                const uint32_t wg = get_write_ptr(cb_g);
                for (uint32_t k = 0; k < CT_STREAM; ++k) {
                    noc_async_read_page(p_pad + k, s, wp + k * TILE_BYTES);
                    noc_async_read_page(g_pad + k, s, wg + k * TILE_BYTES);
                }
                noc_async_read_barrier();
                cb_push_back(cb_p, CT_STREAM);
                cb_push_back(cb_g, CT_STREAM);
            }
        }

        group += group_stride;
        if (group == group_wrap_hi) {
            group = group_wrap_lo;
        }
    }
}
