// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute compute: per input tile, apply the within-tile WH transpose.
//
// Input tile (i, jt) holds x[0, i, jt*32 + col, row] = x[i, jt*32+col, ch] with
// row = ch (channel) within the tile after the transpose. Concretely:
//   in_tile[row, col]                 == x[i, jt*32 + col? ...]
// The standard input tile (i, jt) is x[i, jt*32 + kl, ch] laid out with
// row = (the N-dim sub-index within the i-tile is fixed = il), col index... — see
// reader: page (i, jt) is the [32x32] block x[i fixed-row? ] .  The WH transpose
// turns the tile so that the post-WH tile WHtile[ch, kl] = x[i, jt*32 + kl, ch]
// (row = channel ch, col = j-within-tile kl). This is identical to the proven
// trimul_fused transpose phase (gate removed).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/transpose_wh.h"

#include "../genq/genq_split.h"

void kernel_main() {
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);   // c_0
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);  // c_16 (post-WH)

    // See the dataflow kernels for what GENQ_COMPACT buys. This one needs only the tile COUNT,
    // which is this core's group count times a per-build multiplier (the forward carries Ct
    // channel-tiles a group, the backward one).
    constexpr uint32_t GQ = 2;
    constexpr uint32_t GQ_ON = get_compile_time_arg_val(GQ);
    const genq::Slice gq = genq::slice<
        get_compile_time_arg_val(GQ + 1), get_compile_time_arg_val(GQ + 2),
        get_compile_time_arg_val(GQ + 3), get_compile_time_arg_val(GQ + 4),
        get_compile_time_arg_val(GQ + 5), get_compile_time_arg_val(GQ + 6),
        get_compile_time_arg_val(GQ + 7)>(get_absolute_logical_x(), get_absolute_logical_y());
    const uint32_t num_tiles = GQ_ON ? gq.num * get_compile_time_arg_val(GQ + 8)
                                     : get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;

    transpose_wh_init(in_cb_id, out_cb_id);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(in_cb_id, onetile);
        cb_reserve_back(out_cb_id, onetile);

        transpose_wh_init(in_cb_id, out_cb_id);

        tile_regs_acquire();
        transpose_wh_tile(in_cb_id, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, out_cb_id);
        tile_regs_release();

        cb_pop_front(in_cb_id, onetile);
        cb_push_back(out_cb_id, onetile);
    }
}
