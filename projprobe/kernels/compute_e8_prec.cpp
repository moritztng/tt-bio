// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// E8e -- ONE fp32 tile op, packed out, so its accuracy can be read directly.
// Same three ops as compute_e8_dst.cpp with the accumulation removed, because E8d's residual has to
// be attributed to the arithmetic rather than to running the op twice.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t op = get_compile_time_arg_val(2);

    binary_op_init_common(in_cb, in_cb, out_cb);
    cb_wait_front(in_cb, 1);
    cb_reserve_back(out_cb, 1);

    tile_regs_acquire();
    if constexpr (op == 0) {
        mul_tiles_init(in_cb, in_cb);
        mul_tiles(in_cb, in_cb, 0, 0, 0);
    } else if constexpr (op == 1) {
        add_tiles_init(in_cb, in_cb);
        add_tiles(in_cb, in_cb, 0, 0, 0);
    } else {
        mm_init(in_cb, in_cb, out_cb);
        matmul_tiles(in_cb, in_cb, 0, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);
    cb_pop_front(in_cb, 1);
}
