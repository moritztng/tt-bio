// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// permute_fwd_pk: per unit (ct, it, jt), the 32 row tiles R_i = x[it*32 + i][jt, ct] ([j x c] each):
//   1. transposed in the unpacker ([c x j]) and packed untilized as ONE row of 32 tiles into c_1, so its row c is
//      [R_0^T row c | R_1^T row c | ...] = the [i x j] block of channel c in row-major order, 2 KB contiguous;
//   2. each 2 KB page tilized alone into the output: the channel-major tile of channel ct*32 + c.
// permute_back_pk's mirror with the transpose first, which saves its third pass. A copy through a 16-bit dest:
// bit-exact.
#include <cstdint>
#define REDUCE_OP PoolType::SUM          // tilize.h guards tilize_block on these
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    const uint32_t nunits = get_arg_val<uint32_t>(0);
    constexpr uint32_t in_cb = tt::CBIndex::c_0, rm_cb = tt::CBIndex::c_1, out_cb = tt::CBIndex::c_16;

    compute_kernel_hw_startup(in_cb, rm_cb);
    for (uint32_t u = 0; u < nunits; ++u) {
        transpose_wh_init_short(in_cb);
        pack_untilize_dest_init<8, 32>(rm_cb);
        cb_reserve_back(rm_cb, 32);
        for (uint32_t g = 0; g < 4; ++g) {
            cb_wait_front(in_cb, 8);
            tile_regs_acquire();
            for (uint32_t k = 0; k < 8; ++k) transpose_wh_tile(in_cb, k, k);
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<8, 32>(rm_cb, 1, g);
            tile_regs_release();
            cb_pop_front(in_cb, 8);
        }
        pack_untilize_uninit(rm_cb);
        cb_push_back(rm_cb, 32);

        tilize_init(rm_cb, 1, out_cb);
        for (uint32_t c = 0; c < 32; ++c) {
            cb_wait_front(rm_cb, 1);
            cb_reserve_back(out_cb, 1);
            tilize_block(rm_cb, 1, out_cb);
            cb_push_back(out_cb, 1);
            cb_pop_front(rm_cb, 1);
        }
        tilize_uninit(rm_cb, out_cb);
    }
}
