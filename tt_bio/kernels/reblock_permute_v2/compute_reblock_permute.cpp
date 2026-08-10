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

void kernel_main() {
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);   // c_0
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);  // c_16 (post-WH)

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

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
