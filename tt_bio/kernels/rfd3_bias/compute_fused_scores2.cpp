// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RFD3 fused attention-score compute, L6c: identical arithmetic to compute_fused_scores.cpp, but
// it takes its two operands from ALTERNATING circular-buffer sets because the two data-movement
// RISCs now each build the bias for half the bands (see dm_fused_scores.cpp).
//
// Local band b comes from set ((b + phase_offset) & 1). All four input CBs carry the same two
// formats, so switching sets costs the same unpacker reconfig the two operands already need and
// nothing else. Every parity-relevant line is unchanged from the single-RISC version, and the
// reasons are there: the SFPU add is what ttnn uses for fp32, MUL_UNARY_SFPU is mul_unary_tile
// with the scalar as a bit pattern, and keeping the scaled tile in DST is the identity against
// ttnn's intermediate fp32 pack only while DST is fp32.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_scores_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bias_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scores_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bias_b = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(5);
    constexpr uint32_t Jt = get_compile_time_arg_val(6);
    // Diagnostic only: drop both SFPU passes and pack the bias tile straight through, which prices
    // the two SFPU ops against the rest of the pipeline.
    constexpr uint32_t DIAG_COPY = get_compile_time_arg_val(7);

    const uint32_t num_groups = get_arg_val<uint32_t>(0);
    const uint32_t phase_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_scores_a, cb_bias_a, cb_out);
    add_binary_tile_init();
    binop_with_scalar_tile_init();

    for (uint32_t b = 0; b < num_groups; ++b) {
        const bool second = ((b + phase_offset) & 1) != 0;
        const uint32_t cb_scores = second ? cb_scores_b : cb_scores_a;
        const uint32_t cb_bias = second ? cb_bias_b : cb_bias_a;

        for (uint32_t jt = 0; jt < Jt; ++jt) {
            cb_wait_front(cb_scores, onetile);
            cb_wait_front(cb_bias, onetile);
            cb_reserve_back(cb_out, onetile);

            tile_regs_acquire();
            if constexpr (DIAG_COPY) {
                copy_tile_to_dst_init_short(cb_bias);
                copy_tile(cb_bias, 0, 0);
            } else {
                copy_tile_to_dst_init_short_with_dt(cb_bias, cb_scores);
                copy_tile(cb_scores, 0, 0);
                mul_unary_tile(0, scale_bits);
                copy_tile_to_dst_init_short_with_dt(cb_scores, cb_bias);
                copy_tile(cb_bias, 0, 1);
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();

            cb_pop_front(cb_scores, onetile);
            cb_pop_front(cb_bias, onetile);
            cb_push_back(cb_out, onetile);
        }
    }
}
