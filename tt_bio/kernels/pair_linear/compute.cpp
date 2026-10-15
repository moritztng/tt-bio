// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_linear compute (tt_bio/pair_linear.py): x @ w (+ bias) for one 32-row tile at a time, the
// weight resident in L1. pair_swiglu_ws's fc3 loop alone: per row tile, OG output tiles (one fp32 dest half) at a
// time, each the sum over the KT K tiles in dest, packed bf16 to c_16. PL_BIAS seeds dest with the row-replicated
// bias tiles (c_2) first, as pair_relu does, so the bias is summed in fp32 with the products.
#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t OG = CT < 4 ? CT : 4;
    static_assert(CT % OG == 0);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_out = 16;
#ifdef PL_BIAS
    constexpr uint32_t cb_b = 2;
#endif
    if (count == 0) {
        return;
    }
    mm_block_init(cb_x, cb_w, cb_out, false, OG, 1, KT);
    cb_wait_front(cb_w, KT * CT);
#ifdef PL_BIAS
    cb_wait_front(cb_b, CT);
#endif
    for (uint32_t r = 0; r < count; ++r) {
        cb_wait_front(cb_x, KT);
        for (uint32_t c = 0; c < CT; c += OG) {
            tile_regs_acquire();
#ifdef PL_BIAS
            copy_tile_to_dst_init_short(cb_b);
            reconfig_data_format_srca(cb_b);
            for (uint32_t i = 0; i < OG; ++i) {
                copy_tile(cb_b, c + i, i);
            }
            mm_block_init_short(cb_x, cb_w, false, OG, 1, KT);
            reconfig_data_format(cb_w, cb_x);
#endif
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_x, cb_w, k, k * CT + c, 0, false, OG, 1, KT);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, OG);
            tile_regs_wait();
            pack_tile_block(0, cb_out, OG);
            tile_regs_release();
            cb_push_back(cb_out, OG);
        }
        cb_pop_front(cb_x, KT);
    }
}
