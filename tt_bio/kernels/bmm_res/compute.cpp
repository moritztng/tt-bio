// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// bmm_res compute (tt_bio/bmm_res.py): per channel, a[c] @ b[c] one 32-row tile at a time against the resident
// b[c] (c_1), pair_linear's loop: OG output tiles (one fp32 dest half) at a time, each the sum over the KT K tiles
// in dest, packed to c_16. The whole of K in one dest pass, so the accumulation order is the one in0_block_w = KT
// gives ttnn.matmul.
// TA: each row tile of a arrives untransposed in c_2 (a's format) and is transposed into c_0 (bf16, which holds a
// bf16 or bfp8 value exactly) through dest before its matmuls, ttnn's in0 transpose. srcA carries b for the matmul and
// a for the transpose, so its format is switched both ways. TB: the matmul unpacker transposes each b tile.
#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"

// BRES_DRAIN (bmm_res.DRAIN, a soak knob): every Tensix thread waits for its own issued instructions to finish before
// TA switches the unpacker, math and packer between transpose and matmul, so no reconfiguration overlaps work in flight.
#ifdef BRES_DRAIN
#define BRES_SYNC() do { UNPACK((ckernel::tensix_sync())); MATH((ckernel::tensix_sync())); PACK((ckernel::tensix_sync())); } while (0)
#else
#define BRES_SYNC() do { } while (0)
#endif
// BRES_FULL_TINIT / BRES_FULL_MINIT (bmm_res.SOAK, soak knobs): the TA phase switch runs the full transpose_wh_init /
// mm_block_init (hw configure of every thread, address modes, MOP) instead of the reconfig + short init.

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr bool TA = get_compile_time_arg_val(3);
    constexpr bool TB = get_compile_time_arg_val(4);
    // A split call's raw a arrives in fixed SLOT-tile slots (the reader's AG), the last one of a row part full: a
    // ring of A_DEPTH slots stays aligned only if every push and pop is a whole slot. 0 = unsplit (one push per row).
    constexpr uint32_t SLOT = get_compile_time_arg_val(5);
    constexpr uint32_t BG = get_compile_time_arg_val(6);  // B8: b arrives bf16 in c_3, BG tiles at a time
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t OG = NT % 4 == 0 ? 4 : (NT % 2 == 0 ? 2 : 1);
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_a_raw = 2, cb_b_raw = 3, cb_out = 16;
    if (count == 0) {
        return;
    }
    mm_block_init(cb_a, cb_b, cb_out, TB, OG, 1, KT);
    for (uint32_t ch = 0; ch < count; ++ch) {
        if constexpr (BG > 0) {
            // B8: repack b[c] into the resident c_1 as bfp8 through dest, in arrival order.
            reconfig_data_format_srca(cb_b, cb_b_raw);
            copy_tile_to_dst_init_short(cb_b_raw);
            PACK((pack_reconfig_data_format(cb_b)));
            cb_reserve_back(cb_b, KT * NT);
            for (uint32_t i = 0; i < KT * NT; i += BG) {
                cb_wait_front(cb_b_raw, BG);
                tile_regs_acquire();
                for (uint32_t j = 0; j < BG; ++j) {
                    copy_tile(cb_b_raw, j, j);
                }
                tile_regs_commit();
                cb_pop_front(cb_b_raw, BG);
                tile_regs_wait();
                for (uint32_t j = 0; j < BG; ++j) {
                    pack_tile(j, cb_b);
                }
                tile_regs_release();
            }
            cb_push_back(cb_b, KT * NT);
            reconfig_data_format_srca(cb_b_raw, cb_b);
            mm_block_init_short(cb_a, cb_b, TB, OG, 1, KT);
            PACK((pack_reconfig_data_format(cb_out)));
        }
        cb_wait_front(cb_b, KT * NT);
        for (uint32_t r = 0; r < MT; ++r) {
            if constexpr (TA) {
                BRES_SYNC();
#ifdef BRES_FULL_TINIT
                transpose_wh_init(cb_a_raw, cb_a);
#else
                reconfig_data_format_srca(cb_b, cb_a_raw);
                transpose_wh_init_short(cb_a_raw);
                PACK((pack_reconfig_data_format(cb_a)));
#endif
                for (uint32_t k = 0; k < KT; k += 4) {
                    const uint32_t nb = KT - k < 4 ? KT - k : 4;
                    cb_wait_front(cb_a_raw, nb);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < nb; ++i) {
                        transpose_wh_tile(cb_a_raw, i, i);
                    }
                    tile_regs_commit();
                    cb_pop_front(cb_a_raw, SLOT ? SLOT : nb);
                    cb_reserve_back(cb_a, nb);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < nb; ++i) {
                        pack_tile(i, cb_a);
                    }
                    tile_regs_release();
                    cb_push_back(cb_a, nb);
                }
                BRES_SYNC();
#ifdef BRES_FULL_MINIT
                mm_block_init(cb_a, cb_b, cb_out, TB, OG, 1, KT);
#else
                reconfig_data_format_srca(cb_a_raw, cb_b);
                mm_block_init_short(cb_a, cb_b, TB, OG, 1, KT);
                PACK((pack_reconfig_data_format(cb_out)));
#endif
            }
            cb_wait_front(cb_a, KT);
            for (uint32_t n = 0; n < NT; n += OG) {
                tile_regs_acquire();
#ifndef BRES_NO_MM  // time attribution only (af2_trimul_bmm_sweep --abl)
                for (uint32_t k = 0; k < KT; ++k) {
                    matmul_block(cb_a, cb_b, k, k * NT + n, 0, TB, OG, 1, KT);
                }
#endif
                tile_regs_commit();
                cb_reserve_back(cb_out, OG);
                tile_regs_wait();
                pack_tile_block(0, cb_out, OG);
                tile_regs_release();
                cb_push_back(cb_out, OG);
            }
            cb_pop_front(cb_a, KT);
        }
        cb_pop_front(cb_b, KT * NT);
    }
}
