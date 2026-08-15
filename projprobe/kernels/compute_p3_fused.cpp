// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 3 -- the coordinate stage, the gather and the blend fused into one program.
//
// The novel part is the dataflow direction. §4.4 requires the ADDRESSES to come from the compute
// unit, not the reader: if the reader recomputed floor(xp) itself, one ulp of disagreement would
// fetch a different cell and blend it with the right weights. So this kernel pushes an address tile
// to the gather reader and then waits for the eight corner tiles to come back --
// compute -> reader -> compute, which inverts the usual direction and is the thing this arm exists
// to prove.
//
// Two scratch CBs rather than one. The dense stage pushes 35 intermediates and the blend pushes 26,
// and §8.8 established that a CB used as addressable scratch must have depth exactly equal to what
// one pass pushes -- a larger CB straddles a partial wrap and silently corrupts every block after
// the first. Two exact-sized CBs is the honest way to satisfy that for two different chain lengths.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t cb_e = 0;      // 6 euler tiles per block
constexpr uint32_t cb_xy = 1;     // 2 lattice tiles per block
constexpr uint32_t cb_c = 2;      // 9 constants: the dense 7, then ONE and ODDMASK for the blend
constexpr uint32_t cb_s1 = 3;     // dense scratch, exactly 35
constexpr uint32_t cb_addr = 4;   // the address tile handed to the gather reader
constexpr uint32_t cb_slot = 5;   // the eight gathered corner tiles coming back
constexpr uint32_t cb_s2 = 6;     // blend scratch, exactly 26
constexpr uint32_t cb_out = 16;   // the blended reference, one tile per block

constexpr uint32_t C_MDLXY = 0, C_MDLX = 1, C_ORIGIN = 2, C_SENT = 3, C_SENT23 = 4, C_ONE = 5,
                   C_NEG2 = 6, C_ONE_B = 7, C_ODD = 8;

uint32_t ns;
uint32_t cb_sc;                   // which scratch CB push_s is currently filling

inline void push_s() {
    tile_regs_commit();
    cb_reserve_back(cb_sc, 1);
    tile_regs_wait();
    pack_tile(0, cb_sc);
    tile_regs_release();
    cb_push_back(cb_sc, 1);
    ++ns;
    cb_wait_front(cb_sc, ns);
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

inline void coord(uint32_t ea, uint32_t eb) {
    binop(MUL, cb_e, ea, cb_xy, 0);          push_s();
    binop(MUL, cb_e, eb, cb_xy, 1);          push_s();
    binop(ADD, cb_sc, ns - 2, cb_sc, ns - 1); push_s();
}

// lerp(a, b, f) = a + (b - a) * f, RELION's own association order.
inline uint32_t lerp(uint32_t cba, uint32_t ta, uint32_t cbb, uint32_t tb, uint32_t fslot) {
    binop(SUB, cbb, tb, cba, ta);              push_s();
    binop(MUL, cb_sc, ns - 1, cb_sc, fslot);   push_s();
    binop(ADD, cba, ta, cb_sc, ns - 1);        push_s();
    return ns - 1;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t n_blocks = get_compile_time_arg_val(0);
    const uint32_t r2_lt = get_arg_val<uint32_t>(0);

    init_sfpu(cb_e, cb_out);
    cb_wait_front(cb_c, 9);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_wait_front(cb_e, 6);
        cb_wait_front(cb_xy, 2);

        // ---- the coordinate stage, into cb_s1 ----
        cb_sc = cb_s1;
        ns = 0;

        coord(0, 1);  const uint32_t s_xp = ns - 1;
        coord(2, 3);  const uint32_t s_yp = ns - 1;
        coord(4, 5);  const uint32_t s_zp = ns - 1;

        binop(MUL, cb_s1, s_xp, cb_s1, s_xp);      push_s();
        binop(MUL, cb_s1, s_yp, cb_s1, s_yp);      push_s();
        binop(ADD, cb_s1, ns - 2, cb_s1, ns - 1);  push_s();
        binop(MUL, cb_s1, s_zp, cb_s1, s_zp);      push_s();
        binop(ADD, cb_s1, ns - 2, cb_s1, ns - 1);  push_s();
        const uint32_t s_r2 = ns - 1;

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_s1);
        copy_tile(cb_s1, s_r2, 0);
        unary_lt_tile_init();
        unary_lt_tile(0, r2_lt);
        push_s();
        const uint32_t s_mask = ns - 1;

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_s1);
        copy_tile(cb_s1, s_xp, 0);
        ltz_tile(0);
        push_s();
        binop(MUL, cb_s1, ns - 1, cb_c, C_NEG2);  push_s();
        binop(ADD, cb_s1, ns - 1, cb_c, C_ONE);   push_s();
        const uint32_t s_sgn = ns - 1;

        binop(MUL, cb_s1, s_xp, cb_s1, s_sgn);  push_s();  const uint32_t s_xf = ns - 1;
        binop(MUL, cb_s1, s_yp, cb_s1, s_sgn);  push_s();  const uint32_t s_yf = ns - 1;
        binop(MUL, cb_s1, s_zp, cb_s1, s_sgn);  push_s();  const uint32_t s_zf = ns - 1;

        uint32_t s_fl[3];
        const uint32_t s_f[3] = {s_xf, s_yf, s_zf};
        for (uint32_t d = 0; d < 3; ++d) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_s1);
            copy_tile(cb_s1, s_f[d], 0);
            rounding_op_tile_init();
            floor_tile(0);
            push_s();
            s_fl[d] = ns - 1;
        }
        uint32_t s_fr[3];
        for (uint32_t d = 0; d < 3; ++d) {
            binop(SUB, cb_s1, s_f[d], cb_s1, s_fl[d]);
            push_s();
            s_fr[d] = ns - 1;
        }

        binop(MUL, cb_s1, s_fl[2], cb_c, C_MDLXY);   push_s();
        binop(MUL, cb_s1, s_fl[1], cb_c, C_MDLX);    push_s();
        binop(ADD, cb_s1, ns - 2, cb_s1, ns - 1);    push_s();
        binop(ADD, cb_s1, ns - 1, cb_s1, s_fl[0]);   push_s();
        binop(ADD, cb_s1, ns - 1, cb_c, C_ORIGIN);   push_s();
        binop(SUB, cb_s1, ns - 1, cb_c, C_SENT);     push_s();
        binop(MUL, cb_s1, ns - 1, cb_s1, s_mask);    push_s();
        binop(ADD, cb_s1, ns - 1, cb_c, C_SENT23);   push_s();
        const uint32_t s_addr = ns - 1;
        const uint32_t n_dense = ns;

        // ---- hand the addresses to the gather reader ----
        cb_reserve_back(cb_addr, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_s1);
        copy_tile(cb_s1, s_addr, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_addr);
        tile_regs_release();
        cb_push_back(cb_addr, 1);

        // ---- the blend, into cb_s2, once the corners come back ----
        cb_wait_front(cb_slot, 8);
        cb_sc = cb_s2;
        const uint32_t d_fx = s_fr[0], d_fy = s_fr[1], d_fz = s_fr[2];
        ns = 0;

        // The fractions still live in cb_s1, so the lerps take their f operand from there. lerp()
        // reads f from cb_sc, so the three fraction tiles are copied across first.
        for (uint32_t d = 0; d < 3; ++d) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_s1);
            copy_tile(cb_s1, s_fr[d], 0);
            push_s();
        }
        const uint32_t f_x = 0, f_y = 1, f_z = 2;
        (void)d_fx; (void)d_fy; (void)d_fz;

        const uint32_t dx00 = lerp(cb_slot, 0, cb_slot, 1, f_x);
        const uint32_t dx10 = lerp(cb_slot, 2, cb_slot, 3, f_x);
        const uint32_t dx01 = lerp(cb_slot, 4, cb_slot, 5, f_x);
        const uint32_t dx11 = lerp(cb_slot, 6, cb_slot, 7, f_x);
        const uint32_t dxy0 = lerp(cb_s2, dx00, cb_s2, dx10, f_y);
        const uint32_t dxy1 = lerp(cb_s2, dx01, cb_s2, dx11, f_y);
        const uint32_t ref = lerp(cb_s2, dxy0, cb_s2, dxy1, f_z);

        // sgn and mask are in cb_s1; copy them across so the final ops stay within one scratch CB.
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_s1);
        copy_tile(cb_s1, s_sgn, 0);
        push_s();
        const uint32_t b_sgn = ns - 1;
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_s1);
        copy_tile(cb_s1, s_mask, 0);
        push_s();
        const uint32_t b_mask = ns - 1;

        binop(SUB, cb_s2, b_sgn, cb_c, C_ONE_B);   push_s();
        binop(MUL, cb_s2, ns - 1, cb_c, C_ODD);    push_s();
        binop(ADD, cb_s2, ns - 1, cb_c, C_ONE_B);  push_s();
        const uint32_t sgn_odd = ns - 1;

        binop(MUL, cb_s2, ref, cb_s2, sgn_odd);    push_s();
        binop(MUL, cb_s2, ns - 1, cb_s2, b_mask);  push_s();

        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_s2);
        copy_tile(cb_s2, ns - 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_s2, ns);
        cb_pop_front(cb_s1, n_dense);
        cb_pop_front(cb_slot, 8);
        cb_pop_front(cb_e, 6);
        cb_pop_front(cb_xy, 2);
    }
}
