// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RFD3 fused attention-score compute: out = fp32(scores) * scale + bias, one tile at a time.
//
// This has to be BIT-EXACT against
//     ttnn.add(ttnn.typecast(scores, fp32), bias,
//              input_tensor_a_activations=[UnaryWithParam(MUL_UNARY_SFPU, scale)])
// and every line below is chosen from what that op's own kernels do
// (binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp + eltwise_utils_sfpu.hpp on
// the shipped wheel):
//
//   * ttnn takes the SFPU branch for fp32 operands, so the add is `add_binary_tile`, not the
//     FPU's `add_tiles`. The FPU truncates into DST and would miss (the same trap that made
//     reblock_permute's gate use mul_binary_tile).
//   * the activation is `mul_unary_tile(idst, <fp32 bits>)` -- MUL_UNARY_SFPU emits exactly that,
//     with the scalar as a bit pattern, so the host passes bits and never a float.
//   * ttnn runs the activation as a SEPARATE pass that packs the scaled tile into an fp32
//     intermediate CB and unpacks it again; this kernel keeps it in DST. Those agree only because
//     the intermediate is fp32 -- an fp32 pack is the identity on an fp32 DST value. That is why
//     the descriptor MUST carry fp32_dest_acc_en=True; with 16-bit DST the product would be
//     rounded to bf16 here and to fp32 there.
//   * the bf16 -> fp32 widen ttnn does with a separate typecast happens here in the unpack, which
//     is a 16-bit shift and exact, so no value differs from the one the typecast would have
//     written to DRAM.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_scores = get_compile_time_arg_val(0);  // bf16, from the reader
    constexpr uint32_t cb_bias = get_compile_time_arg_val(1);    // fp32, built by the reader
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);     // fp32, to the writer
    constexpr uint32_t scale_bits = get_compile_time_arg_val(3);
    // Diagnostic only: drop both SFPU passes and pack the bias tile straight through.
    constexpr uint32_t DIAG_COPY = get_compile_time_arg_val(4);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_scores, cb_bias, cb_out);
    add_binary_tile_init();
    binop_with_scalar_tile_init();

    for (uint32_t i = 0; i < num_tiles; ++i) {
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
