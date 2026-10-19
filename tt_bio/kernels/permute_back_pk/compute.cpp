// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// permute_back_pk: per unit (ct, it, jt), the 32 channel tiles T_c = x[c][it, jt] ([i x j] each):
//   1. packed untilized as ONE row of 32 tiles into c_1, so its row i is [T_0 row i | T_1 row i | ...]
//      = the [c x j] block of pair row i in row-major order, 2 KB contiguous;
//   2. each 2 KB page tilized alone into c_2 -> a [c x j] tile;
//   3. transposed in the unpacker -> [j x c], the channel-last output tile of row it*32 + i.
// A pure copy through a 16-bit dest: bit-exact. The same re-tiling dit_pair_bias does, plus the transpose a
// three-axis move needs.
#include <cstdint>
#define REDUCE_OP PoolType::SUM          // tilize.h guards tilize_block on these
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    constexpr uint32_t ABL = get_compile_time_arg_val(0);   // bit 1: no tilize/transpose (diagnostic)
    const uint32_t nunits = get_arg_val<uint32_t>(0);
    constexpr uint32_t in_cb = tt::CBIndex::c_0, rm_cb = tt::CBIndex::c_1, t_cb = tt::CBIndex::c_2;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;

    compute_kernel_hw_startup(in_cb, rm_cb);
    for (uint32_t u = 0; u < nunits; ++u) {
        pack_untilize_init<8, 32>(in_cb, rm_cb);
        cb_reserve_back(rm_cb, 32);
        for (uint32_t g = 0; g < 4; ++g) {
            cb_wait_front(in_cb, 8);
            pack_untilize_block<8, 32>(in_cb, 1, rm_cb, g);
            cb_pop_front(in_cb, 8);
        }
        pack_untilize_uninit(rm_cb);
        cb_push_back(rm_cb, 32);

        if constexpr (ABL & 2) {
            cb_wait_front(rm_cb, 32);
            cb_reserve_back(out_cb, 32);
            cb_push_back(out_cb, 32);
            cb_pop_front(rm_cb, 32);
            continue;
        }
        tilize_init(rm_cb, 1, t_cb);
        for (uint32_t i = 0; i < 32; ++i) {
            cb_wait_front(rm_cb, 1);
            cb_reserve_back(t_cb, 1);
            tilize_block(rm_cb, 1, t_cb);
            cb_push_back(t_cb, 1);
            cb_pop_front(rm_cb, 1);
        }
        tilize_uninit(rm_cb, t_cb);

        // The short init: the full one re-runs pack_sync_init and the hw configure mid-kernel, which hung the chip
        // (bpk, bpk2 on .107); in-tree kernels call it once at start and the short form after other ops.
        transpose_wh_init_short(t_cb);
        for (uint32_t i = 0; i < 32; i += 8) {
            cb_wait_front(t_cb, 8);
            cb_reserve_back(out_cb, 8);
            tile_regs_acquire();
            for (uint32_t k = 0; k < 8; ++k) transpose_wh_tile(t_cb, k, k);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < 8; ++k) pack_tile(k, out_cb);
            tile_regs_release();
            cb_push_back(out_cb, 8);
            cb_pop_front(t_cb, 8);
        }
    }
}
