// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// adaln compute (tt_bio/adaln.py): per row tile, `shift + layer_norm(x) * scale` with the scale and
// shift per token (the DiT's adaLN), where ttnn needs layer_norm then addcmul. Statistics as
// kernels/common/ln_prologue.hpp (two-pass, fp32 CBs, no affine), then * scale and + shift
// elementwise, packed to c_2. fp32 DEST.
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

constexpr uint32_t out_cb = tt::CBIndex::c_2;
constexpr uint32_t cb_sc = tt::CBIndex::c_3;
constexpr uint32_t cb_sb = tt::CBIndex::c_4;
constexpr uint32_t cb_x = tt::CBIndex::c_7;
constexpr uint32_t cb_scaler = tt::CBIndex::c_10;
constexpr uint32_t cb_eps = tt::CBIndex::c_11;
constexpr uint32_t cb_mean = tt::CBIndex::c_12;
constexpr uint32_t cb_var = tt::CBIndex::c_13;
constexpr uint32_t cb_rstd = tt::CBIndex::c_14;
constexpr uint32_t cb_xc = tt::CBIndex::c_16;
constexpr uint32_t cb_tmp = tt::CBIndex::c_17;

struct LnCbs {
    static constexpr uint32_t Kt = ::Kt, x = cb_x, gamma = 0, beta = 0, scaler = cb_scaler,
        eps = cb_eps, mean = cb_mean, var = cb_var, rstd = cb_rstd, xc = cb_xc, tmp = cb_tmp;
};
using namespace ln_prologue;

void kernel_main() {
    const uint32_t nrows = get_arg_val<uint32_t>(0);

    binary_op_init_common(cb_x, cb_scaler, out_cb);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    for (uint32_t r = 0; r < nrows; ++r) {
        row_mean<LnCbs>(cb_x, cb_mean, false);                  // mean
        cb_wait_front(cb_mean, 1);
        stage<LnCbs, SUB_COL>(cb_x, cb_mean, cb_xc);            // xc = x - mean
        cb_pop_front(cb_x, Kt);
        cb_pop_front(cb_mean, 1);
        cb_wait_front(cb_xc, Kt);
        stage<LnCbs, MUL>(cb_xc, cb_xc, cb_tmp);                // xc^2
        row_mean<LnCbs>(cb_tmp, cb_var, true);                  // var (biased, as layer_norm)
        rstd<LnCbs>();                                          // rsqrt(var + eps)
        stage<LnCbs, MUL_COL>(cb_xc, cb_rstd, cb_tmp);          // norm = xc * rstd
        cb_pop_front(cb_xc, Kt);
        cb_pop_front(cb_rstd, 1);
        cb_wait_front(cb_tmp, Kt);
        cb_wait_front(cb_sc, Kt);
        stage<LnCbs, MUL>(cb_tmp, cb_sc, cb_xc);                // * scale
        cb_pop_front(cb_tmp, Kt);
        cb_pop_front(cb_sc, Kt);
        cb_wait_front(cb_xc, Kt);
        cb_wait_front(cb_sb, Kt);
        reconfig_data_format(cb_xc, cb_sb);                     // + shift
        pack_reconfig_data_format(out_cb);
        add_tiles_init(cb_xc, cb_sb);
        cb_reserve_back(out_cb, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            add_tiles(cb_xc, cb_sb, i, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb);
            tile_regs_release();
        }
        cb_push_back(out_cb, Kt);
        cb_pop_front(cb_xc, Kt);
        cb_pop_front(cb_sb, Kt);
    }
}
