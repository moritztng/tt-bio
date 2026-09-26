// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rne_add reader. Two same-shaped bfloat16 TILE tensors, read page-for-page into c_0 and c_1.
//
// A core owns the contiguous tile range [first_tile, first_tile + num_tiles) of BOTH operands --
// the op is elementwise, so one index serves both accessors and the whole work split is a single
// range per core. There is no permutation and no ragged group: the tile count is the padded tile
// count of the tensor, so tile padding is read, added and written like any other tile.
//
// Both addresses are COMMON runtime args because they are the only values that change between
// calls at a fixed shape, which is what lets the host cache the whole ProgramDescriptor and
// rewrite two scalars per call (the same trick `reblock_permute` uses; its header has the cost).
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known device-wedge
// cause in this codebase).
#include "api/dataflow/dataflow_api.h"

#include "../genq/genq_split.h"

void kernel_main() {
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t cb_a = 0;  // c_0
    constexpr uint32_t cb_b = 1;  // c_1
    // Tiles per reserve/push. The CB depth is 2*GRAN, so a block never wraps the CB and the
    // `get_write_ptr` + stride walk below is contiguous.
    constexpr uint32_t GRAN = get_compile_time_arg_val(0);

    constexpr auto a_args = TensorAccessorArgs<1>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto sa = TensorAccessor(a_args, a_addr);
    const auto sb = TensorAccessor(b_args, b_addr);

    // The tile range. Under GENQ_COMPACT it is recomputed from this core's own coordinates so
    // the descriptor carries no per-core runtime args at all; `tt_bio/genq.py` has the price and
    // checks the arithmetic against the host's placement loop before it arms this.
    constexpr uint32_t G = b_args.next_compile_time_args_offset();
    constexpr uint32_t COMPACT = get_compile_time_arg_val(G);
    uint32_t first_tile, num_tiles;
    if constexpr (COMPACT) {
        const auto s = genq::slice<get_compile_time_arg_val(G + 1), get_compile_time_arg_val(G + 2),
                                   get_compile_time_arg_val(G + 3), get_compile_time_arg_val(G + 4),
                                   get_compile_time_arg_val(G + 5), get_compile_time_arg_val(G + 6),
                                   get_compile_time_arg_val(G + 7)>(
            get_absolute_logical_x(), get_absolute_logical_y());
        first_tile = s.first;
        num_tiles = s.num;
    } else {
        first_tile = get_arg_val<uint32_t>(0);
        num_tiles = get_arg_val<uint32_t>(1);
    }

    const uint32_t tile_bytes = get_tile_size(cb_a);

    uint32_t page = first_tile;
    for (uint32_t i = 0; i < num_tiles; i += GRAN) {
        const uint32_t n = (num_tiles - i < GRAN) ? (num_tiles - i) : GRAN;
        cb_reserve_back(cb_a, n);
        cb_reserve_back(cb_b, n);
        uint32_t wa = get_write_ptr(cb_a);
        uint32_t wb = get_write_ptr(cb_b);
        for (uint32_t j = 0; j < n; ++j) {
            // Both streams are in flight together and drained by ONE barrier: the op reads two
            // operands for every one it writes, so serialising them halves the DRAM concurrency
            // this kernel exists to exploit.
            noc_async_read_page(page + j, sa, wa);
            noc_async_read_page(page + j, sb, wb);
            wa += tile_bytes;
            wb += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, n);
        cb_push_back(cb_b, n);
        page += n;
    }
}
