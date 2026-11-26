// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DiffDock fused FC block + tensor product, one core group (tt_bio/diffdock/edge_tile.py has the
// algebra and a float64 emulation of exactly this schedule). Per 32-edge tile:
//   AB[j,kt] = A[kt] * Bfull[j]                     (b_j is a per-edge scalar, so (A@G)*b = (A*b)@G)
//   Y[y]     = sum AB[.] @ Gblock[.]                 (one DST slot per Y tile, matmuls only)
//   per weight tile wi:  Wt = H @ L2[wi]; P[m] = Wt * Y[y_m]; out[ot_m] += P[m] @ onehot[lane_m]
// The last step sums P over its 32 lanes into one output column; the output tiles are fp32 in L1
// and accumulated by the packer (L1 acc), zeroed at the start of each edge tile.
// Schedules arrive as defines: AB_LIST (j_slot, kt) pairs, Y_LIST per Y tile n then n
// (ab, g) pairs, W_LIST per weight tile n then n (y, ot, lane) triples, ot ascending.
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t ab_list[] = {AB_LIST};
constexpr uint32_t y_list[] = {Y_LIST};
constexpr uint32_t w_list[] = {W_LIST};

void kernel_main() {
    constexpr uint32_t KH = get_compile_time_arg_val(0);
    constexpr uint32_t NBJ = get_compile_time_arg_val(1);
    constexpr uint32_t KA = get_compile_time_arg_val(2);
    constexpr uint32_t NAB = get_compile_time_arg_val(3);
    constexpr uint32_t NY = get_compile_time_arg_val(4);
    constexpr uint32_t NW = get_compile_time_arg_val(5);
    constexpr uint32_t NOUT = get_compile_time_arg_val(6);
    constexpr uint32_t NL2 = get_compile_time_arg_val(7);
    constexpr uint32_t NG = get_compile_time_arg_val(8);
    // Diagnostic ablation, 0 in production: bit 0 skips the AB and Y math, bit 1 the weight-tile
    // loop (CB traffic is kept, so a hang that survives an ablation is not in the skipped math).
    constexpr uint32_t ABL = get_compile_time_arg_val(9);
    // Diagnostic tap, 0 in production: 1 writes the Y tiles, 2 the AB tiles, 3 the Wt tiles to
    // the output (fp32, NOUT of them) instead of the contraction.
    constexpr uint32_t TAP = get_compile_time_arg_val(10);
    constexpr uint32_t ZERO = 32;  // the zero tile after the 32 one-hot tiles
    const uint32_t n_et = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_h = 0, cb_a = 1, cb_b = 2, cb_l2 = 3, cb_g = 4, cb_oh = 5;
    constexpr uint32_t cb_ab = 6, cb_y = 7, cb_w = 8, cb_p = 9, cb_out = 16;

    mm_init(cb_h, cb_l2, cb_w);
    cb_wait_front(cb_l2, NL2);
    cb_wait_front(cb_g, NG);
    cb_wait_front(cb_oh, 33);

    for (uint32_t et = 0; et < n_et; ++et) {
        cb_wait_front(cb_h, KH);
        cb_wait_front(cb_a, KA);
        cb_wait_front(cb_b, NBJ);

        // AB = A[kt] * Bfull[j], four per DST window
        mul_tiles_init(cb_a, cb_b);
        pack_reconfig_data_format(cb_ab);
        cb_reserve_back(cb_ab, NAB);
        for (uint32_t i = 0; i < NAB && !(ABL & 1); i += 4) {
            const uint32_t n = NAB - i < 4 ? NAB - i : 4;
            tile_regs_acquire();
            for (uint32_t q = 0; q < n; ++q) mul_tiles(cb_a, cb_b, ab_list[2 * (i + q) + 1], ab_list[2 * (i + q)], q);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t q = 0; q < n; ++q) pack_tile<true>(q, cb_ab, i + q);
            tile_regs_release();
        }
        cb_push_back(cb_ab, NAB);
        cb_pop_front(cb_a, KA);
        cb_pop_front(cb_b, NBJ);
        cb_wait_front(cb_ab, NAB);

        // Y tiles
        mm_init_short(cb_ab, cb_g);
        pack_reconfig_data_format(cb_y);
        cb_reserve_back(cb_y, NY);
        uint32_t p = 0;
        for (uint32_t yi = 0; yi < NY && !(ABL & 1); ++yi) {
            const uint32_t n = y_list[p++];
            tile_regs_acquire();
            for (uint32_t q = 0; q < n; ++q, p += 2) matmul_tiles(cb_ab, cb_g, y_list[p], y_list[p + 1], 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_y, yi);
            tile_regs_release();
        }
        cb_push_back(cb_y, NY);
        cb_wait_front(cb_y, NY);
        if constexpr (TAP != 0) {
            constexpr uint32_t src = TAP == 1 ? cb_y : (TAP == 2 ? cb_ab : cb_l2);
            cb_reserve_back(cb_out, NOUT);
            pack_reconfig_data_format(cb_out);
            for (uint32_t o = 0; o < NOUT; ++o) {
                if constexpr (TAP == 3) {
                    mm_init_short(cb_h, cb_l2);
                } else {
                    copy_tile_to_dst_init_short(src);
                }
                tile_regs_acquire();
                if constexpr (TAP == 3) {
                    for (uint32_t kt = 0; kt < KH; ++kt) matmul_tiles(cb_h, cb_l2, kt, o * KH + kt, 0);
                } else {
                    copy_tile(src, o, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_out, o);
                tile_regs_release();
            }
            cb_push_back(cb_out, NOUT);
            cb_pop_front(cb_ab, NAB);
            cb_pop_front(cb_h, KH);
            cb_pop_front(cb_y, NY);
            continue;
        }
        cb_pop_front(cb_ab, NAB);

        // zero the output tiles
        cb_reserve_back(cb_out, NOUT);
        copy_tile_to_dst_init_short(cb_oh);
        pack_reconfig_data_format(cb_out);
        pack_reconfig_l1_acc(0);
        tile_regs_acquire();
        for (uint32_t o = 0; o < NOUT; ++o) copy_tile(cb_oh, ZERO, o);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t o = 0; o < NOUT; ++o) pack_tile<true>(o, cb_out, o);
        tile_regs_release();

        p = 0;
        for (uint32_t wi = 0; wi < NW && !(ABL & 2); ++wi) {
            const uint32_t n = w_list[p];
            const uint32_t* e = &w_list[p + 1];
            p += 1 + 3 * n;
            // Wt = H @ L2[wi]
            mm_init_short(cb_h, cb_l2);
            pack_reconfig_data_format(cb_w);
            pack_reconfig_l1_acc(0);
            cb_reserve_back(cb_w, 1);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < KH; ++kt) matmul_tiles(cb_h, cb_l2, kt, wi * KH + kt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_w);
            tile_regs_release();
            cb_push_back(cb_w, 1);
            cb_wait_front(cb_w, 1);
            // P[m] = Wt * Y[y_m]
            mul_tiles_init(cb_w, cb_y);
            pack_reconfig_data_format(cb_p);
            cb_reserve_back(cb_p, n);
            tile_regs_acquire();
            for (uint32_t q = 0; q < n; ++q) mul_tiles(cb_w, cb_y, 0, e[3 * q], q);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t q = 0; q < n; ++q) pack_tile<true>(q, cb_p, q);
            tile_regs_release();
            cb_push_back(cb_p, n);
            cb_pop_front(cb_w, 1);
            cb_wait_front(cb_p, n);
            // out[ot] += P[m] @ onehot[lane], one DST slot per distinct output tile
            const uint32_t ot0 = e[1];
            const uint32_t nslot = e[3 * (n - 1) + 1] - ot0 + 1;
            mm_init_short(cb_p, cb_oh);
            pack_reconfig_data_format(cb_out);
            pack_reconfig_l1_acc(1);
            tile_regs_acquire();
            for (uint32_t q = 0; q < n; ++q) matmul_tiles(cb_p, cb_oh, q, e[3 * q + 2], e[3 * q + 1] - ot0);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t s = 0; s < nslot; ++s) pack_tile<true>(s, cb_out, ot0 + s);
            tile_regs_release();
            cb_pop_front(cb_p, n);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(cb_out, NOUT);
        cb_pop_front(cb_h, KH);
        cb_pop_front(cb_y, NY);
    }
}
