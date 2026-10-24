// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// permute_back_pk: per unit (ct, it, jt), the 32 channel tiles T_c = x[c][it, jt] ([i x j] each):
//   1. packed untilized as ONE row of 32 tiles into c_1, so its row i is [T_0 row i | T_1 row i | ...]
//      = the [c x j] block of pair row i in row-major order, 2 KB contiguous;
//   2. each 2 KB page tilized alone into c_2 -> a [c x j] tile;
//   3. transposed in the unpacker -> [j x c], the channel-last output tile of row it*32 + i.
// TR = 0 (tile_rows) stops after step 2: the tilized page is the output tile.
// X8 = 1: x is bfp8 (RFD4's `pair_x8`); the unpacker widens it to bf16 in step 1, steps 2-3 read bf16 as before.
// A pure copy through a 16-bit dest: bit-exact against x as stored. The same re-tiling dit_pair_bias does, plus the transpose a
// three-axis move needs.
#include <cstdint>
#define REDUCE_OP PoolType::SUM          // tilize.h guards tilize_block on these
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t ABL = get_compile_time_arg_val(0);   // bit 1: no tilize/transpose, bit 3: no transpose (diagnostic)
    constexpr uint32_t TR = get_compile_time_arg_val(1);    // 1: transpose each tilized tile, 0: write it as is
    constexpr uint32_t X8 = get_compile_time_arg_val(2);    // 1: in_cb is bfp8, so srcA switches format per stage
    const uint32_t nunits = get_arg_val<uint32_t>(0);
    constexpr uint32_t in_cb = tt::CBIndex::c_0, rm_cb = tt::CBIndex::c_1, t_cb = tt::CBIndex::c_2;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;

    compute_kernel_hw_startup(in_cb, rm_cb);
    for (uint32_t u = 0; u < nunits; ++u) {
        // copy_tile + pack_untilize_dest, as dit_pair_bias and trimul_gin_moved's RETILE do on device;
        // pack_untilize_block (no in-tree user) was the first suspect when bpk3 still hung.
        if constexpr (X8) reconfig_data_format_srca(in_cb);
        copy_tile_to_dst_init_short(in_cb);
        pack_untilize_dest_init<8, 32>(rm_cb);
        cb_reserve_back(rm_cb, 32);
        for (uint32_t g = 0; g < 4; ++g) {
            cb_wait_front(in_cb, 8);
            tile_regs_acquire();
            for (uint32_t k = 0; k < 8; ++k) copy_tile(in_cb, k, k);
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<8, 32>(rm_cb, 1, g);
            tile_regs_release();
            cb_pop_front(in_cb, 8);
        }
        pack_untilize_uninit(rm_cb);
        cb_push_back(rm_cb, 32);

        if constexpr (ABL & 2) {
            // In 8s: out_cb holds 16 tiles, so reserving 32 at once never returns (that hung bpk4).
            for (uint32_t i = 0; i < 32; i += 8) {
                cb_wait_front(rm_cb, 8);
                cb_reserve_back(out_cb, 8);
                cb_push_back(out_cb, 8);
                cb_pop_front(rm_cb, 8);
            }
            continue;
        }
        constexpr bool tr = TR && !(ABL & 8);
        constexpr uint32_t tz_cb = tr ? t_cb : out_cb;
        if constexpr (X8) reconfig_data_format_srca(rm_cb);
        tilize_init(rm_cb, 1, tz_cb);
        for (uint32_t i = 0; i < 32; ++i) {
            cb_wait_front(rm_cb, 1);
            cb_reserve_back(tz_cb, 1);
            tilize_block(rm_cb, 1, tz_cb);
            cb_push_back(tz_cb, 1);
            cb_pop_front(rm_cb, 1);
        }
        tilize_uninit(rm_cb, tz_cb);
        if constexpr (!tr) continue;

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
