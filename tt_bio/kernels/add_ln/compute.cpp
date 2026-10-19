// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_ln compute (tt_bio/add_ln.py): per row tile, h = a + b (FPU add into the fp32 DEST, packed to bf16, which is
// ttnn.add's answer element for element: kernels/rne_add/compute_rne_add.cpp), packed twice, to c_2 for the writer
// and to c_7 for the norm; then layer_norm(h) (kernels/common/ln_prologue.hpp, two-pass fp32 statistics) packed to
// c_3 for the writer.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "../common/ln_prologue.hpp"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t BLK = 4;   // fp32 half-sync DST holds 4 tiles

constexpr uint32_t cb_a = tt::CBIndex::c_0;
constexpr uint32_t cb_b = tt::CBIndex::c_1;
constexpr uint32_t cb_h = tt::CBIndex::c_2;
constexpr uint32_t cb_y = tt::CBIndex::c_3;

struct LnCbs {
    static constexpr uint32_t Kt = ::Kt, x = tt::CBIndex::c_7, gamma = tt::CBIndex::c_8, beta = tt::CBIndex::c_9,
        scaler = tt::CBIndex::c_10, eps = tt::CBIndex::c_11, mean = tt::CBIndex::c_12, var = tt::CBIndex::c_13,
        rstd = tt::CBIndex::c_14, xc = tt::CBIndex::c_16, tmp = tt::CBIndex::c_17;
};

void kernel_main() {
    const uint32_t nrows = get_arg_val<uint32_t>(0);
    binary_op_init_common(cb_a, cb_b, cb_h);
    cb_wait_front(LnCbs::gamma, Kt);
    cb_wait_front(LnCbs::beta, Kt);
    cb_wait_front(LnCbs::scaler, 1);
    cb_wait_front(LnCbs::eps, 1);

    for (uint32_t r = 0; r < nrows; ++r) {
        cb_wait_front(cb_a, Kt);
        cb_wait_front(cb_b, Kt);
        cb_reserve_back(cb_h, Kt);
        cb_reserve_back(LnCbs::x, Kt);
        reconfig_data_format(cb_a, cb_b);
        pack_reconfig_data_format(cb_h);
        add_tiles_init(cb_a, cb_b);
        for (uint32_t i0 = 0; i0 < Kt; i0 += BLK) {
            const uint32_t n = Kt - i0 < BLK ? Kt - i0 : BLK;
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) add_tiles(cb_a, cb_b, i0 + j, i0 + j, j);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile<true>(j, cb_h, i0 + j);
                pack_tile<true>(j, LnCbs::x, i0 + j);
            }
            tile_regs_release();
        }
        cb_push_back(cb_h, Kt);
        cb_push_back(LnCbs::x, Kt);
        cb_pop_front(cb_a, Kt);
        cb_pop_front(cb_b, Kt);

        cb_reserve_back(cb_y, Kt);
        ln_prologue::layer_norm_row<LnCbs>(cb_y, [](uint32_t i) { pack_tile<true>(0, cb_y, i); });
        cb_push_back(cb_y, Kt);
    }
}
