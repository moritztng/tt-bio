// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_build compute: z = ((const + g) * s) * wg + beta per tile, every operand unpacked straight to a 32-bit DEST and
// every op the SFPU's float32 binary, in the order the ttnn ops it replaces ran them (tt_bio.rfd4.pair.PairOnCard).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

static_assert(DST_ACCUM_MODE == 1, "pair_build needs a 32-bit DEST (fp32_dest_acc_en)");

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t KT = get_compile_time_arg_val(2);
    constexpr uint32_t cb_c = 0, cb_g = 1, cb_s = 2, cb_wg = 3, cb_beta = 4, cb_out = 16;
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    init_sfpu(cb_c, cb_out);
    uint32_t design = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t b = u / NT / N;
        if (b != design) {
            if (design != 0xFFFFFFFFu) {
                cb_pop_front(cb_wg, KT);
                cb_pop_front(cb_beta, KT);
            }
            cb_wait_front(cb_wg, KT);
            cb_wait_front(cb_beta, KT);
            design = b;
        }
        cb_wait_front(cb_s, 1);
        cb_wait_front(cb_c, KT);
        cb_wait_front(cb_g, KT);
        cb_reserve_back(cb_out, KT);
        for (uint32_t ct = 0; ct < KT; ++ct) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_c);
            copy_tile(cb_c, ct, 0);
            copy_tile_to_dst_init_short(cb_g);
            copy_tile(cb_g, ct, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            copy_tile_to_dst_init_short(cb_s);
            copy_tile(cb_s, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            copy_tile_to_dst_init_short(cb_wg);
            copy_tile(cb_wg, ct, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            copy_tile_to_dst_init_short(cb_beta);
            copy_tile(cb_beta, ct, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_pop_front(cb_s, 1);
        cb_pop_front(cb_c, KT);
        cb_pop_front(cb_g, KT);
        cb_push_back(cb_out, KT);
    }
    if (design != 0xFFFFFFFFu) {
        cb_pop_front(cb_wg, KT);
        cb_pop_front(cb_beta, KT);
    }
}
