// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// E8c -- price every primitive the coarse projection kernel's op budget is written in.
//
// §4.3 of state/relion-kernel-coarse-projection.md counts the kernel in "tile ops" and E8 measured
// exactly one of them, mul_tiles, at 61.10 ns in fp32. The design also leans on transpose_wh_tile
// (once per gathered tile), on the bcast_rows family (the primitive that reconciles dense per-pair
// math with the gather's [pair, slot] layout) and on reduce_tile in both directions (the blend sum
// and the pixel sum). If any of those costs several mul_tiles the budget is wrong before a line of
// the real kernel is written, which is what P2 pre-registers: each of them <= 3 FPU tile ops.
//
// Same loop shape as compute_e4_blend.cpp so every number subtracts against E8's own baseline: one
// tile in, `ops` back-to-back primitives into DST, one pack, one tile out, `outer` times. `prim`
// selects which primitive, so there is one kernel rather than eight near-copies.
//
//   0 mul_tiles (the E8 baseline, re-measured here as the control)
//   1 transpose_wh_tile
//   2 reduce_tile<SUM, REDUCE_COL>
//   3 reduce_tile<SUM, REDUCE_ROW>
//   4 mul_tiles_bcast_rows
//   5 add_tiles_bcast_rows
//   6 mul_tiles_bcast_cols
//   7 matmul_tiles
//   8 trunc_tile   (SFPU; the radius test's rounding)
//   9 frac_tile    (SFPU; would replace floor+subtract if it costs the same as floor)
//  10 mul_binary_tile  (SFPU, DST-to-DST)
//  11 add_binary_tile  (SFPU, DST-to-DST)
//  12 sub_binary_tile  (SFPU, DST-to-DST)
//  13 reduce_tile<SUM, REDUCE_COL, enforce_fp32_accumulation=true>
//  14 unary_bcast<ROW>   15 unary_bcast<COL>  -- bcast.h implements these via unpack-to-dest for
//     32-bit formats, on its own initiative ("SrcB is only 19bits wide"), so unlike the binary
//     bcast family they can be exact in fp32. That makes them the only precision-safe broadcast on
//     this hardware, and §4.3's layout reconciliation depends on having one.
//
// 10-12 are the ones that matter now. E8g showed every FPU op truncates its operand to ~11 mantissa
// bits and that only the SFPU, with unpack_to_dest, holds fp32 exactly. So the coarse kernel's
// numeric path has to be built from these three, and §4.3's budget -- written in FPU tile ops --
// has to be re-priced in them.
#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_COL)

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t ops = get_compile_time_arg_val(2);
    constexpr uint32_t outer = get_compile_time_arg_val(3);
    constexpr uint32_t sc_cb = get_compile_time_arg_val(4);
    constexpr uint32_t prim = get_compile_time_arg_val(5);

    constexpr bool is_sfpu = (prim >= 8 && prim <= 12);   // 13 is a reduce, which is FPU
    constexpr bool is_sfpu_binary = (prim >= 10 && prim <= 12);

    if constexpr (is_sfpu) {
        init_sfpu(in_cb, out_cb);
    } else {
        binary_op_init_common(in_cb, in_cb, out_cb);
    }
    cb_wait_front(in_cb, 1);

    // The cycle has to be complete -- acquire/commit without the matching wait/pack/release hangs
    // the device (compute_e4_blend.cpp's own note). A real kernel packs its tile anyway.
    if (ops > 0) {
        for (uint32_t i = 0; i < outer; ++i) {
            cb_reserve_back(sc_cb, 1);
            tile_regs_acquire();
            if constexpr (is_sfpu_binary) {
                copy_tile_to_dst_init_short(in_cb);
                copy_tile(in_cb, 0, 0);
                copy_tile(in_cb, 0, 1);
                if constexpr (prim == 10) {
                    mul_binary_tile_init();
                } else if constexpr (prim == 11) {
                    add_binary_tile_init();
                } else {
                    sub_binary_tile_init();
                }
                for (uint32_t k = 0; k < ops; ++k) {
                    if constexpr (prim == 10) {
                        mul_binary_tile(0, 1, 0);
                    } else if constexpr (prim == 11) {
                        add_binary_tile(0, 1, 0);
                    } else {
                        sub_binary_tile(0, 1, 0);
                    }
                }
            } else if constexpr (prim == 14 || prim == 15) {
                constexpr auto bt = (prim == 14) ? BroadcastType::ROW : BroadcastType::COL;
                unary_bcast_init<bt>(in_cb, sc_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    unary_bcast<bt>(in_cb, 0, 0);
                }
            } else if constexpr (prim == 13) {
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL, true>(in_cb, in_cb, sc_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL, true>(in_cb, in_cb, 0, 0, 0);
                }
                reduce_uninit<true>();
            } else if constexpr (is_sfpu) {
                copy_tile_to_dst_init_short(in_cb);
                copy_tile(in_cb, 0, 0);
                rounding_op_tile_init();
                for (uint32_t k = 0; k < ops; ++k) {
                    if constexpr (prim == 8) {
                        trunc_tile(0);
                    } else {
                        frac_tile(0);
                    }
                }
            } else if constexpr (prim == 0) {
                mul_tiles_init(in_cb, in_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    mul_tiles(in_cb, in_cb, 0, 0, 0);
                }
            } else if constexpr (prim == 1) {
                transpose_wh_init_short(in_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    transpose_wh_tile(in_cb, 0, 0);
                }
            } else if constexpr (prim == 2) {
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(in_cb, in_cb, sc_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(in_cb, in_cb, 0, 0, 0);
                }
                reduce_uninit();
            } else if constexpr (prim == 3) {
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(in_cb, in_cb, sc_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(in_cb, in_cb, 0, 0, 0);
                }
                reduce_uninit();
            } else if constexpr (prim == 4) {
                mul_bcast_rows_init_short(in_cb, in_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    mul_tiles_bcast_rows(in_cb, in_cb, 0, 0, 0, k & 31);
                }
            } else if constexpr (prim == 5) {
                add_bcast_rows_init_short(in_cb, in_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    add_tiles_bcast_rows(in_cb, in_cb, 0, 0, 0, k & 31);
                }
            } else if constexpr (prim == 6) {
                mul_bcast_cols_init_short(in_cb, in_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    mul_tiles_bcast_cols(in_cb, in_cb, 0, 0, 0);
                }
            } else {
                mm_init(in_cb, in_cb, sc_cb);
                for (uint32_t k = 0; k < ops; ++k) {
                    matmul_tiles(in_cb, in_cb, 0, 0, 0);
                }
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
