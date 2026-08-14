// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// E4c -- does the arithmetic hide under the gather?
//
// E4/E4b priced the gather alone at 18.96 ns per 16 B read on two RISCs, which puts RELION's coarse
// E-step at 9.37 s on one p150. That is a floor only if the math the kernel also has to do rides
// along for free. It has to do two things per output pixel: blend 8 corners trilinearly (8 weighted
// multiply-accumulates per complex component) and then form (ref-shifted)^2 summed over pixels and
// scaled by the CTF weight. Both are tile-wise once the reader has placed each gathered corner pair
// into the right slot, which it can: `noc_async_read` chooses its own L1 destination, so the CB
// comes out already tile-shaped and no transpose or shuffle is needed.
//
// So the shape of the math is `ops` back-to-back tile multiply-accumulates, and this kernel runs
// that many, `outer` times, against a CB the reader pushes UP FRONT rather than at the end. The
// gather loop and this loop are then live at the same time on the same core. Sweeping `ops` from 0
// finds the point where the wall departs from the gather-only 1213 ns, which is where the math stops
// hiding.
//
// The tiles it reads are being overwritten by the gather underneath it. That is deliberate and it is
// a timing screen, not an arithmetic one: the question is whether the math unit's occupancy adds to
// the dataflow RISCs' wall, and the answer does not depend on the values.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t ops = get_compile_time_arg_val(2);
    constexpr uint32_t outer = get_compile_time_arg_val(3);
    constexpr uint32_t sc_cb = get_compile_time_arg_val(4);

    binary_op_init_common(in_cb, in_cb, out_cb);
    cb_wait_front(in_cb, 1);

    // The cycle has to be complete. tile_regs_acquire/commit are MATH-thread macros and
    // tile_regs_wait/release are PACK-thread ones, so acquire+commit+release without the wait and
    // the pack leaves the packer signalling dest-section-done for a section it never waited on.
    // Measured: ops=0 ran fine (the loop is skipped) and ops=8 hung the device on the first call.
    // A real kernel packs its blended tile anyway, so the honest loop is the whole cycle.
    if (ops > 0) {
        for (uint32_t i = 0; i < outer; ++i) {
            cb_reserve_back(sc_cb, 1);
            tile_regs_acquire();
            mul_tiles_init(in_cb, in_cb);
            for (uint32_t k = 0; k < ops; ++k) {
                // mul_tiles ACCUMULATES into DST, which is exactly a weighted corner blend.
                mul_tiles(in_cb, in_cb, 0, 0, 0);
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
