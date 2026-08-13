// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// S2b -- the matmul roof, measured rather than asserted.
//
// S1b's matmul arm gave a 50.37 ns marginal cost per 32x32x32 tile matmul, which is 1.31 TFLOP/s
// per core and 170 TFLOP/s across 130 cores. That is above this card's published bf16 matmul rate,
// so it cannot be a true fp32 number and something in the measurement is being reused. The obvious
// candidate is the operand pattern: S1b called matmul_tiles(cb, cb, 0, 1, dst) every time, so srcA
// and srcB never changed and the unpacker may have skipped reloading them.
//
// `walk` is the fix. With walk set, each of the K accumulating matmuls takes a different pair of
// tiles out of an NT-deep CB, which is what a real output-block accumulation does. The gap between
// walk=0 and walk=1 is the size of the reuse effect, and walk=1 is the number the FFT budget is
// allowed to use.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);   // distinct tiles resident in in_cb
    constexpr uint32_t walk = get_compile_time_arg_val(5);

    const uint32_t outer = get_arg_val<uint32_t>(0);
    constexpr uint32_t onetile = 1;

    binary_op_init_common(in_cb, in_cb, scratch_cb);
    cb_wait_front(in_cb, NT);

    for (uint32_t i = 0; i < outer; ++i) {
        cb_reserve_back(scratch_cb, onetile);
        tile_regs_acquire();
        mm_init(in_cb, in_cb, scratch_cb, 0);
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t a = walk ? (k % NT) : 0;
            const uint32_t b = walk ? ((k + NT / 2) % NT) : 1;
            matmul_tiles(in_cb, in_cb, a, b, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, scratch_cb);
        tile_regs_release();
        cb_push_back(scratch_cb, onetile);
        cb_wait_front(scratch_cb, onetile);
        cb_pop_front(scratch_cb, onetile);
    }

    cb_reserve_back(out_cb, onetile);
    tile_regs_acquire();
    mm_init(in_cb, in_cb, out_cb, 0);
    matmul_tiles(in_cb, in_cb, 0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, onetile);
    cb_pop_front(in_cb, NT);
}
