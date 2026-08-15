// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// E8b -- price an SFPU tile op against the FPU tile op E8 measured.
//
// The coarse projection kernel's weight side is not all eltwise binary: it needs floor and frac per
// coordinate to split xp into x0 and fx, and those run on the SFPU over the DST register rather than
// on the FPU. E4c's free budget was measured with `mul_tiles` only, so an SFPU op that costs several
// FPU ops would quietly move the assembled kernel from gather-bound to compute-bound. Same loop
// shape as compute_e4_blend.cpp so the two numbers subtract.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t ops = get_compile_time_arg_val(2);
    constexpr uint32_t outer = get_compile_time_arg_val(3);
    constexpr uint32_t sc_cb = get_compile_time_arg_val(4);

    init_sfpu(in_cb, out_cb);
    cb_wait_front(in_cb, 1);

    if (ops > 0) {
        for (uint32_t i = 0; i < outer; ++i) {
            cb_reserve_back(sc_cb, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, 0, 0);
            rounding_op_tile_init();
            for (uint32_t k = 0; k < ops; ++k) {
                floor_tile(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, sc_cb);
            tile_regs_release();
            cb_push_back(sc_cb, 1);
            cb_wait_front(sc_cb, 1);
            cb_pop_front(sc_cb, 1);
        }
    }

    cb_reserve_back(out_cb, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(in_cb);
    copy_tile(in_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);
    cb_pop_front(in_cb, 1);
}
