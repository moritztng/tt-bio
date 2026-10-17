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

#include "../genq/genq_split.h"

void kernel_main() {
    // The cheap dispatch path. Under GENQ_COMPACT this kernel's slice of the group split is
    // recomputed from its own logical coordinates, so the descriptor carries no per-core runtime
    // args and the dispatch costs a third of what it costs with them (`tt_bio/genq.py`). The
    // shape words and the walk constants are compile-time: with WALK="block" -- the only mode the
    // host arms this for -- `_walk` returns (block, 1, NO_WRAP, 0), and `block` IS the slice.
    //
    // The block sits FIRST in the compile-time args, at a fixed offset, because these kernels
    // read their runtime args above the line that declares the tensor accessor.
    constexpr uint32_t GQ = 0;
    constexpr uint32_t GQ_ON = get_compile_time_arg_val(GQ);
    const genq::Slice gq = genq::slice<
        get_compile_time_arg_val(GQ + 1), get_compile_time_arg_val(GQ + 2),
        get_compile_time_arg_val(GQ + 3), get_compile_time_arg_val(GQ + 4),
        get_compile_time_arg_val(GQ + 5), get_compile_time_arg_val(GQ + 6),
        get_compile_time_arg_val(GQ + 7)>(get_absolute_logical_x(), get_absolute_logical_y());

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    // The destination may be a wider [1, N, N, Ct_out*32] tensor that this call fills only the
    // channel tiles [ct_off, ct_off + Ct) of, so a channel loop writes its chunks in place and
    // never concatenates them. Ct_out == Ct and ct_off == 0 is the plain move.
    const uint32_t Ct_out = get_common_arg_val<uint32_t>(1);
    const uint32_t ct_off = get_common_arg_val<uint32_t>(2);
    // Logical N. When it is not a multiple of 32 the last row group (it = Nt-1) owns only
    // N - it*32 real output rows i: the rest would land past the end of the outer axis, so they
    // are dropped here (the reader and compute still run the fixed 32 tiles a group). And the
    // last column group (jt = Nt-1) carries rows jl >= N - jt*32 of every output tile, which are
    // tile padding on the j axis: zeroed, as the forward writer does, because a consumer that
    // contracts over j reads them.
    const uint32_t N = get_common_arg_val<uint32_t>(4);
    const uint32_t first_group = (GQ_ON ? (gq.first) : get_arg_val<uint32_t>(0));
    const uint32_t num_groups = (GQ_ON ? (gq.num) : get_arg_val<uint32_t>(1));
    const uint32_t Nt = (GQ_ON ? (get_compile_time_arg_val(GQ + 8)) : get_arg_val<uint32_t>(2));
    const uint32_t Ct = (GQ_ON ? (get_compile_time_arg_val(GQ + 10)) : get_arg_val<uint32_t>(3));
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
    const uint32_t group_stride = (GQ_ON ? (1u) : get_arg_val<uint32_t>(4));
    const uint32_t group_wrap_hi = (GQ_ON ? (0xFFFFFFFFu) : get_arg_val<uint32_t>(5));
    const uint32_t group_wrap_lo = (GQ_ON ? (0u) : get_arg_val<uint32_t>(6));

    constexpr uint32_t element_size = get_compile_time_arg_val(11);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(12);     // c_16
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(13);   // 32
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(14);    // 32
    constexpr auto dst_args = TensorAccessorArgs<4 + 11>();



    constexpr uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * element_size;  // 2048

    const auto s = TensorAccessor(dst_args, dst_addr);
#ifdef RPB_SPLIT
    // RPB_SPLIT (reblock_permute.BACK_SPLIT): this RISC also reads each group's 32 channel-major source tiles from
    // DRAM into the stage ring, one group ahead, on its own NOC, so the read overlaps the reader's L1 gather of the
    // group before. The reader only gathers.
    constexpr uint32_t stage_cb = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    constexpr auto src_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset() + 1>();
    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(3));
    auto stage = [&](uint32_t g) {
        const uint32_t it = g / (Nt * Ct), rem = g - it * (Nt * Ct), jt = rem / Ct, ct = rem - jt * Ct;
        cb_reserve_back(stage_cb, TILE_HEIGHT);
        uint32_t page = (ct * TILE_HEIGHT) * (Nt * Nt) + it * Nt + jt, dst = get_write_ptr(stage_cb);
        for (uint32_t cl = 0; cl < TILE_HEIGHT; ++cl) {
            noc_async_read_page(page, src, dst);
            page += Nt * Nt;
            dst += tile_bytes;
        }
    };
    auto next_group = [&](uint32_t g) {
        g += group_stride;
        return g == group_wrap_hi ? group_wrap_lo : g;
    };
    uint32_t ahead = first_group;
    if (num_groups) {
        stage(ahead);
        noc_async_read_barrier();
        cb_push_back(stage_cb, TILE_HEIGHT);
    }
#endif

    const uint32_t NtCt = Nt * Ct;
    const uint32_t NtCt_out = Nt * Ct_out;
    uint32_t group = first_group;
    for (uint32_t gi = 0; gi < num_groups; ++gi) {
        const uint32_t it = group / NtCt;
        const uint32_t rem = group - it * NtCt;
        const uint32_t jt = rem / Ct;
        const uint32_t ct = rem - jt * Ct;

#ifdef RPB_SPLIT
        // Stage the next group before waiting on this one's output, so the reader can gather it as soon as it is done.
        if (gi + 1 < num_groups) {
            ahead = next_group(ahead);
            stage(ahead);
            noc_async_read_barrier();
            cb_push_back(stage_cb, TILE_HEIGHT);
        }
#endif
        cb_wait_front(cb_id_out, TILE_HEIGHT);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        uint32_t out_page = (it * TILE_HEIGHT) * NtCt_out + jt * Ct_out + ct + ct_off;
        const uint32_t rows_i = (it * TILE_HEIGHT + TILE_HEIGHT <= N) ? TILE_HEIGHT : (N - it * TILE_HEIGHT);
        const uint32_t rows_j = (jt * TILE_HEIGHT + TILE_HEIGHT <= N) ? TILE_HEIGHT : (N - jt * TILE_HEIGHT);
        if (rows_j < TILE_HEIGHT) {
            // Output tile row jl sits in face pair jl/16, row jl%16 of each; one face row is 16 elements.
            constexpr uint32_t FACE = 16, FACE_ROW_WORDS = FACE * element_size / 4;
            constexpr uint32_t FACE_ELEMS = FACE * FACE;
            for (uint32_t il = 0; il < rows_i; ++il) {
                const uint32_t tb = l1_read_addr + il * tile_bytes;
                for (uint32_t jl = rows_j; jl < TILE_HEIGHT; ++jl) {
                    for (uint32_t fw = 0; fw < 2; ++fw) {
                        const uint32_t e = ((jl / FACE) * 2 + fw) * FACE_ELEMS + (jl % FACE) * FACE;
                        volatile tt_l1_ptr uint32_t* z =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb + e * element_size);
                        for (uint32_t k = 0; k < FACE_ROW_WORDS; ++k) {
                            z[k] = 0;
                        }
                    }
                }
            }
        }
        for (uint32_t il = 0; il < rows_i; ++il) {
#ifndef RPB_NO_WRITE  // bench ablation (reblock_permute.BACK_ABL & 4)
            noc_async_write(l1_read_addr, s.get_noc_addr(out_page), tile_bytes);
#endif
            l1_read_addr += tile_bytes;
            out_page += NtCt_out;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, TILE_HEIGHT);

        group += group_stride;
        if (group == group_wrap_hi) {
            group = group_wrap_lo;
        }
    }
}
