// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 -- the squared-difference compare and the pixel accumulation.
//
// RELION's diff2 body, for one orientation block and one translation:
//     d2 = sum_pixels ( (ref_r - sh_r)^2 + (ref_i - sh_i)^2 ) * w
// In the paired-column layout each pixel owns two adjacent columns, re and im, so the real and
// imaginary differences are the even and odd columns of ONE tile and their squares are one tile op.
// Summing over both columns is exactly the |.|^2 the formula asks for -- the component sum comes out
// of the layout rather than out of an extra op.
//
// Four SFPU ops per pixel block:
//     d   = ref - sh
//     d2  = d * d
//     wd  = d2 * w
//     acc = acc + wd            (element-wise across pixel blocks, and exact)
//
// The accumulation across pixel blocks is element-wise, so it stays on the exact SFPU set. What is
// left afterwards is a fold of the 32 columns of `acc` into one number per orientation, which no
// element-wise op can do; that fold is 6 orientation blocks x 9 translations per call, i.e. tiny,
// and is deliberately NOT done here -- see §8.12.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t cb_ref = 0;    // the blended reference, one tile per pixel block
constexpr uint32_t cb_sh = 1;     // the shift stack for this translation, same shape
constexpr uint32_t cb_w = 2;      // the per-pixel weight, replicated across each pair's two columns
constexpr uint32_t cb_s = 3;      // scratch
constexpr uint32_t cb_out = 16;   // the accumulator, emitted once at the end

uint32_t ns;

inline void push_s() {
    tile_regs_commit();
    cb_reserve_back(cb_s, 1);
    tile_regs_wait();
    pack_tile(0, cb_s);
    tile_regs_release();
    cb_push_back(cb_s, 1);
    ++ns;
    cb_wait_front(cb_s, ns);
}

enum Op { MUL, ADD, SUB };

inline void binop(Op op, uint32_t cba, uint32_t ta, uint32_t cbb, uint32_t tb) {
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cba);
    copy_tile(cba, ta, 0);
    copy_tile_to_dst_init_short(cbb);
    copy_tile(cbb, tb, 1);
    if (op == MUL) {
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
    } else if (op == ADD) {
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    } else {
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t n_blocks = get_compile_time_arg_val(0);

    init_sfpu(cb_ref, cb_out);
    ns = 0;

    uint32_t acc = 0;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_wait_front(cb_ref, 1);
        cb_wait_front(cb_sh, 1);
        cb_wait_front(cb_w, 1);

        binop(SUB, cb_ref, 0, cb_sh, 0);          push_s();
        binop(MUL, cb_s, ns - 1, cb_s, ns - 1);   push_s();
        binop(MUL, cb_s, ns - 1, cb_w, 0);        push_s();
        if (b == 0) {
            acc = ns - 1;
        } else {
            binop(ADD, cb_s, acc, cb_s, ns - 1);  push_s();
            acc = ns - 1;
        }

        cb_pop_front(cb_ref, 1);
        cb_pop_front(cb_sh, 1);
        cb_pop_front(cb_w, 1);
    }

    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_s);
    copy_tile(cb_s, acc, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_s, ns);
}
