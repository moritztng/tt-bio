// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// bmm_res compute (tt_bio/bmm_res.py): per channel, a[c] @ b[c] one 32-row tile at a time against the resident
// b[c] (c_1), pair_linear's loop: OG output tiles (one fp32 dest half) at a time, each the sum over the KT K tiles
// in dest, packed to c_16. The whole of K in one dest pass, so the accumulation order is the one in0_block_w = KT
// gives ttnn.matmul.
#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t OG = NT % 4 == 0 ? 4 : (NT % 2 == 0 ? 2 : 1);
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_out = 16;
    if (count == 0) {
        return;
    }
    mm_block_init(cb_a, cb_b, cb_out, false, OG, 1, KT);
    for (uint32_t ch = 0; ch < count; ++ch) {
        cb_wait_front(cb_b, KT * NT);
        for (uint32_t r = 0; r < MT; ++r) {
            cb_wait_front(cb_a, KT);
            for (uint32_t n = 0; n < NT; n += OG) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < KT; ++k) {
                    matmul_block(cb_a, cb_b, k, k * NT + n, 0, false, OG, 1, KT);
                }
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
