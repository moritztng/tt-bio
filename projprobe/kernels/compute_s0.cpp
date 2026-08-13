// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S0 -- re-verify on card 0 the four per-tile-op rates the section-6 budget rests on. The FFT spike
// measured them on cards 2 and 3; the budget is 0.243 us/slice against a 0.312 us/slice DRAM write
// floor, so a 20% error in the eltwise rate flips the kernel from write-bound to compute-bound and
// the prediction has to be rewritten. Cheap to re-run, expensive to be wrong about.
//
// The rate is read off the SLOPE in K so the fixed per-iteration cost -- the L1 round trip, the CB
// push/pop, the loop -- differences out and does not inflate the roof. `mode` picks the arm;
// operands walk an NT-deep CB in every arm so no arm can be measuring unpacker reuse.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

#define MODE_MATMUL 0
#define MODE_MUL    1
#define MODE_ADD    2
#define MODE_COPY   3

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);
    constexpr uint32_t mode = get_compile_time_arg_val(5);

    const uint32_t outer = get_arg_val<uint32_t>(0);
    constexpr uint32_t onetile = 1;

    binary_op_init_common(in_cb, in_cb, scratch_cb);
    cb_wait_front(in_cb, NT);

    for (uint32_t i = 0; i < outer; ++i) {
        cb_reserve_back(scratch_cb, onetile);
        tile_regs_acquire();
        if constexpr (mode == MODE_MATMUL) {
            mm_init(in_cb, in_cb, scratch_cb, 0);
            for (uint32_t k = 0; k < K; ++k) {
                matmul_tiles(in_cb, in_cb, k % NT, (k + NT / 2) % NT, 0);
            }
        } else if constexpr (mode == MODE_MUL) {
            mul_tiles_init(in_cb, in_cb);
            for (uint32_t k = 0; k < K; ++k) {
                mul_tiles(in_cb, in_cb, k % NT, (k + NT / 2) % NT, k % 4);
            }
        } else if constexpr (mode == MODE_ADD) {
            add_tiles_init(in_cb, in_cb);
            for (uint32_t k = 0; k < K; ++k) {
                add_tiles(in_cb, in_cb, k % NT, (k + NT / 2) % NT, k % 4);
            }
        } else {
            copy_tile_to_dst_init_short(in_cb);
            for (uint32_t k = 0; k < K; ++k) {
                copy_tile(in_cb, k % NT, k % 4);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, scratch_cb);
        tile_regs_release();
        cb_push_back(scratch_cb, onetile);
        cb_wait_front(scratch_cb, onetile);
        cb_pop_front(scratch_cb, onetile);
    }

    // Make the result observable so no arm's arithmetic can have been elided.
    cb_reserve_back(out_cb, onetile);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(in_cb);
    copy_tile(in_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, onetile);
    cb_pop_front(in_cb, NT);
}
