// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// E8d -- does an eltwise binary op accumulate into DST, or overwrite it?
//
// compute_e4_blend.cpp asserts in a comment that "mul_tiles ACCUMULATES into DST, which is exactly a
// weighted corner blend". That kernel was a timing screen and says so: "the answer does not depend on
// the values". The coarse projection kernel's dense stage does depend on it. If eltwise binary
// accumulates, xp = e0*x + e1*y is two mul_tiles into one DST slot and the whole coordinate stage
// collapses; if it overwrites, every sum needs its own pack-and-re-read cycle, or a matmul, whose
// header does promise DST += C.
//
// So: two mul_tiles of the same tile into DST slot 0, packed out. Accumulate gives 2*x*x, overwrite
// gives x*x. `op` picks mul (0) or add (1), because the two need not behave the same.
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
        mul_tiles(in_cb, in_cb, 0, 0, 0);
    } else if constexpr (op == 1) {
        add_tiles_init(in_cb, in_cb);
        add_tiles(in_cb, in_cb, 0, 0, 0);
        add_tiles(in_cb, in_cb, 0, 0, 0);
    } else {
        // The control: matmul_tiles documents DST += C, so two of them must give 2*(A@B).
        mm_init(in_cb, in_cb, out_cb);
        matmul_tiles(in_cb, in_cb, 0, 0, 0);
        matmul_tiles(in_cb, in_cb, 0, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);
    cb_pop_front(in_cb, 1);
}
