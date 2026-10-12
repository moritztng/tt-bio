// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_ln_transpose compute: each of a unit's 32 row tiles normalised over its Kt channel tiles
// (kernels/common/ln_prologue.hpp, two-pass fp32 statistics as ttnn.layer_norm) and packed to slot
// r * Kt + i of the unit's 32 * Kt tiles in c_16, which the writer transposes.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "../common/ln_prologue.hpp"

struct LnCbs {
    static constexpr uint32_t Kt = get_compile_time_arg_val(0), x = 5, gamma = 8, beta = 9, scaler = 10, eps = 11,
                              mean = 12, var = 13, rstd = 14, xc = 17, tmp = 18;
};

void kernel_main() {
    constexpr uint32_t Kt = LnCbs::Kt;
    constexpr uint32_t cb_n = 16;
    const uint32_t nunits = get_arg_val<uint32_t>(0);
    if (nunits == 0) {
        return;
    }
    binary_op_init_common(LnCbs::x, LnCbs::scaler, cb_n);
    cb_wait_front(LnCbs::gamma, Kt);
    cb_wait_front(LnCbs::beta, Kt);
    cb_wait_front(LnCbs::scaler, 1);
    cb_wait_front(LnCbs::eps, 1);
    for (uint32_t u = 0; u < nunits; ++u) {
        cb_reserve_back(cb_n, 32 * Kt);
        for (uint32_t r = 0; r < 32; ++r) {
            ln_prologue::layer_norm_row<LnCbs>(cb_n, [r](uint32_t i) { pack_tile<true>(0, cb_n, r * Kt + i); });
        }
        cb_push_back(cb_n, 32 * Kt);
    }
}
