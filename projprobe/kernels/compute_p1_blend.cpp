// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1(iii) -- the trilinear blend. Eight gathered corners and three fractions in, one reference
// tile out, entirely in exact fp32.
//
// RELION's own cascade, transcribed unchanged (acc_projectorkernel_impl.h, and re-transcribed in
// tt_bio/cryoem/relion.py:_project):
//     dx00 = lerp(c000, c100, fx)   dx10 = lerp(c010, c110, fx)
//     dx01 = lerp(c001, c101, fx)   dx11 = lerp(c011, c111, fx)
//     dxy0 = lerp(dx00, dx10, fy)   dxy1 = lerp(dx01, dx11, fy)
//     ref  = lerp(dxy0, dxy1, fz)
// with lerp(a, b, f) = a + (b - a) * f -- RELION's form, not the algebraically equal
// a*(1-f) + b*f, because the two differ in the last bits and §6's gate is a residual against
// RELION's own answer.
//
// Seven lerps, three SFPU ops each. Every one is a DST-to-DST binary under unpack_to_dest, the only
// set §8.4 measured exact (0.000e+00); every FPU alternative truncates the operand to ~11 mantissa
// bits, and this stage multiplies model values by weights, so that error would land straight on
// diff2.
//
// The paired-column layout (§8.7 correction) does the complex bookkeeping for free: each pixel owns
// two adjacent columns, re and im, so the same weight multiplies both and the seven lerps are
// ordinary tile ops with no notion of complexity in them. Only the Friedel conjugate distinguishes
// the two columns, and that is one multiply against a tile that is 1 on even columns and sgn on odd:
//     sgn_odd = 1 + (sgn - 1) * oddmask
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t cb_slot = 0;   // 8 gathered corner tiles
constexpr uint32_t cb_den = 1;    // fx fy fz mask sgn
constexpr uint32_t cb_c = 2;      // ONE, ODDMASK
constexpr uint32_t cb_s = 3;      // scratch
constexpr uint32_t cb_out = 16;   // one reference tile

constexpr uint32_t D_FX = 0, D_FY = 1, D_FZ = 2, D_MASK = 3, D_SGN = 4;
constexpr uint32_t C_ONE = 0, C_ODD = 1;

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

// lerp(a, b, f) = a + (b - a) * f, in RELION's own association order.
inline uint32_t lerp(uint32_t cba, uint32_t ta, uint32_t cbb, uint32_t tb, uint32_t f) {
    binop(SUB, cbb, tb, cba, ta);        push_s();       // b - a
    binop(MUL, cb_s, ns - 1, cb_den, f); push_s();       // (b - a) * f
    binop(ADD, cba, ta, cb_s, ns - 1);   push_s();       // a + that
    return ns - 1;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t n_blocks = get_compile_time_arg_val(0);

    init_sfpu(cb_slot, cb_out);
    cb_wait_front(cb_c, 2);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_wait_front(cb_slot, 8);
        cb_wait_front(cb_den, 5);
        ns = 0;

        const uint32_t dx00 = lerp(cb_slot, 0, cb_slot, 1, D_FX);
        const uint32_t dx10 = lerp(cb_slot, 2, cb_slot, 3, D_FX);
        const uint32_t dx01 = lerp(cb_slot, 4, cb_slot, 5, D_FX);
        const uint32_t dx11 = lerp(cb_slot, 6, cb_slot, 7, D_FX);
        const uint32_t dxy0 = lerp(cb_s, dx00, cb_s, dx10, D_FY);
        const uint32_t dxy1 = lerp(cb_s, dx01, cb_s, dx11, D_FY);
        const uint32_t ref = lerp(cb_s, dxy0, cb_s, dxy1, D_FZ);

        // sgn_odd = 1 + (sgn - 1) * oddmask: the Friedel conjugate negates the imaginary part only,
        // and in this layout the imaginary part is simply the odd columns.
        binop(SUB, cb_den, D_SGN, cb_c, C_ONE);   push_s();
        binop(MUL, cb_s, ns - 1, cb_c, C_ODD);    push_s();
        binop(ADD, cb_s, ns - 1, cb_c, C_ONE);    push_s();
        const uint32_t sgn_odd = ns - 1;

        binop(MUL, cb_s, ref, cb_s, sgn_odd);     push_s();
        binop(MUL, cb_s, ns - 1, cb_den, D_MASK); push_s();

        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_s);
        copy_tile(cb_s, ns - 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_s, ns);
        cb_pop_front(cb_slot, 8);
        cb_pop_front(cb_den, 5);
    }
}
