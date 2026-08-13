// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S1b -- the kill gate for the fused-FFT thesis.
//
// The L1-round-count model in the design doc rests on one assumption: that inside a fused kernel,
// arithmetic performed on tiles already resident in the DST register is effectively free, so a
// stage costs one L1 round trip rather than one round trip per arithmetic operation. Screen S1
// measured 0.351 TFLOP/s for ttnn-op-granularity eltwise and that number bounds a COMPOSITE
// implementation only, because every ttnn op pays unpack + pack per tile. This kernel measures the
// fused case directly.
//
// Two fp32 tiles are loaded into c_0 once, before the loop. Each iteration performs exactly one L1
// read round (the unpack) and one L1 write round (the pack), with K arithmetic operations between
// them. Flat in K means arithmetic is free inside a round and the model holds; linear in K means
// every operation costs a round and no FFT on this card can approach the DRAM floor.
//
// The scratch CB is reserved, packed, pushed, waited on and popped inside the loop rather than
// reserved once outside it. Packing repeatedly into a slot reserved once hangs all 130 cores:
// cb_reserve_back is what advances the packer's own credit accounting, so without it the PACK
// thread blocks after the first tile and the program never retires. Push and pop are pointer
// arithmetic and move no data, so the loop is still one read round and one write round; the cost
// they add is identical across arms and cancels in the K sweep.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh_dest.h"

// Arms. Each isolates one engine, so the ns-per-iteration curve names its own mechanism.
#define ARM_SFPU_DEST 0    // K x mul_binary_tile, DST-to-DST, arithmetic touches no L1
#define ARM_FPU_MUL 1      // K x mul_tiles, each re-unpacking both operands from L1
#define ARM_FPU_MATMUL 2   // K x matmul_tiles 32x32x32 -- this is screen S2
#define ARM_TRANSPOSE 3    // K x transpose_wh_dest, the in-place DST transpose
#define ARM_COPY_ONLY 4    // the round-trip floor: unpack 2, pack 1, no arithmetic

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);    // the writer's CB, pushed once
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(2);  // the loop's pack target
    constexpr uint32_t arm = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t is_32bit = get_compile_time_arg_val(5);

    const uint32_t outer = get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;

    binary_op_init_common(in_cb, in_cb, scratch_cb);

    cb_wait_front(in_cb, 2);

    for (uint32_t i = 0; i < outer; ++i) {
        cb_reserve_back(scratch_cb, onetile);
        tile_regs_acquire();

        if constexpr (arm == ARM_SFPU_DEST || arm == ARM_COPY_ONLY) {
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, 0, 0);
            copy_tile(in_cb, 1, 1);
            if constexpr (arm == ARM_SFPU_DEST) {
                mul_binary_tile_init();
                for (uint32_t k = 0; k < K; ++k) {
                    mul_binary_tile(0, 1, 0);
                }
            }
        } else if constexpr (arm == ARM_FPU_MUL) {
            mul_tiles_init(in_cb, in_cb);
            for (uint32_t k = 0; k < K; ++k) {
                mul_tiles(in_cb, in_cb, 0, 1, 0);
            }
        } else if constexpr (arm == ARM_FPU_MATMUL) {
            mm_init(in_cb, in_cb, scratch_cb, 0);
            for (uint32_t k = 0; k < K; ++k) {
                matmul_tiles(in_cb, in_cb, 0, 1, 0);
            }
        } else if constexpr (arm == ARM_TRANSPOSE) {
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, 0, 0);
            transpose_wh_dest_init_short<is_32bit != 0>();
            for (uint32_t k = 0; k < K; ++k) {
                transpose_wh_dest<is_32bit != 0>(0);
            }
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, scratch_cb);
        tile_regs_release();
        cb_push_back(scratch_cb, onetile);

        // Drain immediately: the scratch CB is the loop's own sink, so nothing downstream consumes
        // it and it would otherwise fill and block.
        cb_wait_front(scratch_cb, onetile);
        cb_pop_front(scratch_cb, onetile);
    }

    // One real output tile so the result is observable and the arithmetic cannot have been elided.
    cb_reserve_back(out_cb, onetile);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(in_cb);
    copy_tile(in_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, onetile);

    cb_pop_front(in_cb, 2);
}
