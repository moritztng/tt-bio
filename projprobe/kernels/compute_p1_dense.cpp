// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1(i) -- the dense coordinate stage of RELION's exact axis-aligned trilinear projection.
//
// Per (32 orientations x 32 pixels) block this computes, for every (orientation, pixel) pair:
//   xp = e0*x + e1*y,  yp = e3*x + e4*y,  zp = e6*x + e7*y      (padding_factor = 1)
//   inside = (xp^2 + yp^2 + zp^2) < maxR2_padded + 1            (RELION truncs the float sum and
//                                                                compares <= maxR2_padded, and for a
//                                                                non-negative sum those are the same
//                                                                test -- one SFPU op instead of two)
//   Friedel: sgn = 1 - 2*[xp < 0], applied to all three coordinates
//   x0 = floor(xp), fx = xp - x0, likewise y and z
//   addr = z0*mdlXY + y0*mdlX + x0 + origin, emitted as float(addr) + 2^23 so the reader can take
//          the raw word and mask the low 23 bits; outside the radius it emits the sentinel 2^23-1
//
// EVERY arithmetic op here is an SFPU DST-to-DST op under unpack_to_dest. That is not a stylistic
// choice: E8e/E8g measured every FPU path truncating its operand to about 11 mantissa bits (an fp32
// add_tiles is 6.99e-4, and a+a is exact arithmetic in any format), while the SFPU set under
// unpack_to_dest came back at exactly 0.000e+00 against torch fp32. An 11-bit xp is fatal here
// rather than merely inaccurate: |xp| reaches 98, so a 2e-3 relative error is 0.2 absolute, and fx
// -- a fraction in [0, 1) -- would be wrong by 0.2.
//
// The layout follows §8.7: everything is a dense [orientation, pixel] tile, so no operand ever needs
// a broadcast, which matters because E8k found no precision-safe broadcast exists on this hardware.
// The host supplies e0..e7 already replicated across columns and x/y already replicated down rows;
// both are reused across the whole call, so building them costs nothing per pair.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t cb_e = 0;    // 6 tiles per orientation block: e0 e1 e3 e4 e6 e7
constexpr uint32_t cb_xy = 1;   // 2 tiles per pixel block: x y
constexpr uint32_t cb_c = 2;    // 7 constant tiles, pushed once and never popped
constexpr uint32_t cb_s = 3;    // scratch, one slot per intermediate
constexpr uint32_t cb_out = 16; // 6 tiles per block: addr fx fy fz mask sgn

// Constant tile indices in cb_c.
constexpr uint32_t C_MDLXY = 0, C_MDLX = 1, C_ORIGIN = 2, C_SENT = 3, C_SENT23 = 4, C_ONE = 5,
                   C_NEG2 = 6;

uint32_t ns;  // how many tiles have been pushed to cb_s, so slot k is readable once ns > k

// Pack DST slot 0 into the next cb_s slot and make it readable.
inline void push_s() {
    tile_regs_commit();
    cb_reserve_back(cb_s, 1);
    tile_regs_wait();
    pack_tile(0, cb_s);
    tile_regs_release();
    cb_push_back(cb_s, 1);
    ++ns;
    cb_wait_front(cb_s, ns);   // slots 0..ns-1 stay readable; nothing is popped until the block ends
}

// dst_out = a OP b, where a and b name (cb, tile) pairs. One acquire, two copies, one SFPU op.
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

// One coordinate: e_a * x + e_b * y, two multiplies and an add, all on the SFPU.
inline void coord(uint32_t ea, uint32_t eb) {
    binop(MUL, cb_e, ea, cb_xy, 0);
    push_s();                                   // e_a * x
    binop(MUL, cb_e, eb, cb_xy, 1);
    push_s();                                   // e_b * y
    binop(ADD, cb_s, ns - 2, cb_s, ns - 1);
    push_s();                                   // the coordinate
}

}  // namespace

void kernel_main() {
    constexpr uint32_t n_blocks = get_compile_time_arg_val(0);
    const uint32_t r2_lt = get_arg_val<uint32_t>(0);   // float bits of maxR2_padded + 1

    init_sfpu(cb_e, cb_out);
    cb_wait_front(cb_c, 7);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_wait_front(cb_e, 6);
        cb_wait_front(cb_xy, 2);
        ns = 0;

        coord(0, 1);  const uint32_t s_xp = ns - 1;
        coord(2, 3);  const uint32_t s_yp = ns - 1;
        coord(4, 5);  const uint32_t s_zp = ns - 1;

        // r2 = xp^2 + yp^2 + zp^2
        binop(MUL, cb_s, s_xp, cb_s, s_xp);   push_s();
        binop(MUL, cb_s, s_yp, cb_s, s_yp);   push_s();
        binop(ADD, cb_s, ns - 2, cb_s, ns - 1);  push_s();
        binop(MUL, cb_s, s_zp, cb_s, s_zp);   push_s();
        binop(ADD, cb_s, ns - 2, cb_s, ns - 1);  push_s();
        const uint32_t s_r2 = ns - 1;

        // mask = r2 < maxR2_padded + 1
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_s);
        copy_tile(cb_s, s_r2, 0);
        unary_lt_tile_init();
        unary_lt_tile(0, r2_lt);
        push_s();
        const uint32_t s_mask = ns - 1;

        // sgn = 1 - 2*[xp < 0]
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_s);
        copy_tile(cb_s, s_xp, 0);
        ltz_tile(0);
        push_s();
        binop(MUL, cb_s, ns - 1, cb_c, C_NEG2);  push_s();
        binop(ADD, cb_s, ns - 1, cb_c, C_ONE);   push_s();
        const uint32_t s_sgn = ns - 1;

        // The Friedel-flipped coordinates, then their floors and fractions.
        binop(MUL, cb_s, s_xp, cb_s, s_sgn);  push_s();  const uint32_t s_xf = ns - 1;
        binop(MUL, cb_s, s_yp, cb_s, s_sgn);  push_s();  const uint32_t s_yf = ns - 1;
        binop(MUL, cb_s, s_zp, cb_s, s_sgn);  push_s();  const uint32_t s_zf = ns - 1;

        uint32_t s_fl[3];
        const uint32_t s_f[3] = {s_xf, s_yf, s_zf};
        for (uint32_t d = 0; d < 3; ++d) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_s);
            copy_tile(cb_s, s_f[d], 0);
            rounding_op_tile_init();
            floor_tile(0);
            push_s();
            s_fl[d] = ns - 1;
        }
        uint32_t s_fr[3];
        for (uint32_t d = 0; d < 3; ++d) {
            binop(SUB, cb_s, s_f[d], cb_s, s_fl[d]);
            push_s();
            s_fr[d] = ns - 1;
        }

        // addr = z0*mdlXY + y0*mdlX + x0 + origin, masked to the sentinel outside the radius, and
        // biased by 2^23 so its mantissa IS the integer.
        binop(MUL, cb_s, s_fl[2], cb_c, C_MDLXY);      push_s();
        binop(MUL, cb_s, s_fl[1], cb_c, C_MDLX);       push_s();
        binop(ADD, cb_s, ns - 2, cb_s, ns - 1);        push_s();
        binop(ADD, cb_s, ns - 1, cb_s, s_fl[0]);       push_s();
        binop(ADD, cb_s, ns - 1, cb_c, C_ORIGIN);      push_s();
        binop(SUB, cb_s, ns - 1, cb_c, C_SENT);        push_s();
        binop(MUL, cb_s, ns - 1, cb_s, s_mask);        push_s();
        binop(ADD, cb_s, ns - 1, cb_c, C_SENT23);      push_s();
        const uint32_t s_addr = ns - 1;

        const uint32_t out[6] = {s_addr, s_fr[0], s_fr[1], s_fr[2], s_mask, s_sgn};
        for (uint32_t k = 0; k < 6; ++k) {
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_s);
            copy_tile(cb_s, out[k], 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }

        cb_pop_front(cb_s, ns);
        cb_pop_front(cb_e, 6);
        cb_pop_front(cb_xy, 2);
    }
}
