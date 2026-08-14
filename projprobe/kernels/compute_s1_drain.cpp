// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S1's compute kernel does nothing but move the reader's one assembled tile to the output CB, so
// the assembled bytes reach DRAM and the arm can be checked for having actually read something. An
// arm whose reads could have been elided is not a measurement of those reads.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);

    binary_op_init_common(in_cb, in_cb, out_cb);
    cb_wait_front(in_cb, 1);
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
