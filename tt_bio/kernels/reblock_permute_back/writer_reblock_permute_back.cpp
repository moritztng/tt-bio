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

    const uint32_t NtCt = Nt * Ct;
    uint32_t group = first_group;
    for (uint32_t gi = 0; gi < num_groups; ++gi) {
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

        group += group_stride;
        if (group == group_wrap_hi) {
            group = group_wrap_lo;
        }
    }
}
