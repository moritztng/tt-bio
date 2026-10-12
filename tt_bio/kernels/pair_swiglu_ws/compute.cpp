// SPDX-License-Identifier: Apache-2.0
//
// pair_swiglu_ws: fc3(silu(x @ w1) * (x @ w2)) for one 32-row tile at a time, weights resident in L1.
//
// Per row tile, for each group of SW hidden tiles: fc1 group -> silu -> c_3 (gate); fc2 group, times the gate in
// dest (dest reused as srcB, as kernels/fc12g does) -> c_4 (hidden). With all NH hidden tiles of the row in c_4,
// fc3 accumulates the CT output tiles in dest and packs them to c_16. x is read once, the output written once,
// and fc1/fc2/the product never leave the core.
//
// Draft: every CB is bf16 (no unpack/pack reconfig between phases), SW = CT = 4 (one fp32 dest half).
#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t SW = 4;
    static_assert(NH % SW == 0 && CT <= 4);
    constexpr uint32_t cb_x = 0, cb_w12 = 1, cb_w3 = 2, cb_gate = 3, cb_hid = 4, cb_out = 16;
#if !defined(TRISC_MATH)
    constexpr auto mul_fidelity = MathFidelity::HiFi4;
#else
    constexpr auto mul_fidelity = static_cast<MathFidelity>(MATH_FIDELITY);
#endif
    if (count == 0) {
        return;
    }
    mm_block_init(cb_x, cb_w12, cb_gate, false, SW, 1, KT);
    cb_wait_front(cb_w12, KT * 2 * NH);
    cb_wait_front(cb_w3, NH * CT);

    for (uint32_t r = 0; r < count; ++r) {
        cb_wait_front(cb_x, KT);
        for (uint32_t j = 0; j < NH; j += SW) {
            mm_block_init_short_with_dt(cb_x, cb_w12, cb_gate, false, SW, 1, KT);
            // fc1 group -> silu -> gate
            tile_regs_acquire();
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_x, cb_w12, k, k * 2 * NH + j, 0, false, SW, 1, KT);
            }
            silu_tile_init();  // the dest-reuse multiply's init does not keep the SFPU's
            for (uint32_t i = 0; i < SW; ++i) {
                silu_tile(i);
            }
            tile_regs_commit();
            cb_reserve_back(cb_gate, SW);
            tile_regs_wait();
            pack_tile_block(0, cb_gate, SW);
            tile_regs_release();
            cb_push_back(cb_gate, SW);
            // fc2 group, times the gate in dest -> hidden
            tile_regs_acquire();
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_x, cb_w12, k, k * 2 * NH + NH + j, 0, false, SW, 1, KT);
            }
            UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                false, false, cb_gate)));
            MATH((llk_math_eltwise_binary_init<ELWMUL, NONE, mul_fidelity, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                false)));
            cb_wait_front(cb_gate, SW);
            for (uint32_t i = 0; i < SW; ++i) {
                UNPACK((llk_unpack_A<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_gate, i)));
                MATH((llk_math_eltwise_binary<ELWMUL, NONE, DST_ACCUM_MODE, mul_fidelity,
                                              EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_gate, cb_gate, i, true)));
            }
            cb_pop_front(cb_gate, SW);
            tile_regs_commit();
            cb_reserve_back(cb_hid, SW);
            tile_regs_wait();
            pack_tile_block(0, cb_hid, SW);
            tile_regs_release();
            cb_push_back(cb_hid, SW);
        }
        cb_pop_front(cb_x, KT);
        // fc3 over the row's NH hidden tiles
        mm_block_init_short_with_dt(cb_hid, cb_w3, cb_gate, false, CT, 1, NH);
        cb_wait_front(cb_hid, NH);
        tile_regs_acquire();
        for (uint32_t k = 0; k < NH; ++k) {
            matmul_block(cb_hid, cb_w3, k, k * CT, 0, false, CT, 1, NH);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, CT);
        tile_regs_wait();
        pack_tile_block(0, cb_out, CT);
        tile_regs_release();
        cb_push_back(cb_out, CT);
        cb_pop_front(cb_hid, NH);
    }
}
