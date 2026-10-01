// Layer-norm backward over the last dim, one tile-row (32 rows x K) at a time, all in L1:
//   xc = x - mean(x); rstd = rsqrt(mean(xc^2) + eps); norm = xc * rstd; dn = g * gamma
//   dx = (dn - mean(dn) - norm * mean(dn * norm)) * rstd
// The same two-pass statistics as `autograd._layer_norm_bw`; intermediates are float32 CBs and
// the reductions accumulate in a float32 DEST.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t DO_GAMMA = get_compile_time_arg_val(2);
constexpr uint32_t cb_x = 0, cb_g = 1, cb_gamma = 2, cb_scaler = 3, cb_eps = 4, cb_mean = 5,
                   cb_xc = 6, cb_tmp = 7, cb_var = 8, cb_rstd = 9, cb_norm = 10, cb_dn = 11,
                   cb_a = 12, cb_b = 13, cb_t = 14, cb_out = 16;

// Sum of each row of Wt tiles of `in`, times the 1/K scaler, into one tile of `out`.
ALWI void row_mean(uint32_t in, uint32_t out, bool pop) {
    reconfig_data_format(in, cb_scaler);
    pack_reconfig_data_format(out);
    cb_wait_front(in, Wt);
    cb_reserve_back(out, 1);
    tile_regs_acquire();
    reduce_init<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, out);
    for (uint32_t i = 0; i < Wt; ++i) reduce_tile<REDUCE_OP, REDUCE_DIM, true>(in, cb_scaler, i, 0, 0);
    reduce_uninit<true>();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    if (pop) cb_pop_front(in, Wt);
}

enum Op { SUB_COL, MUL_COL, MUL_ROW, MUL, SUB };

// out[i] = a[i] (op) b[i or 0], Wt tiles. Neither input is popped here.
template <Op OP>
ALWI void stage(uint32_t a, uint32_t b, uint32_t out) {
    reconfig_data_format(a, b);
    pack_reconfig_data_format(out);
    if constexpr (OP == SUB_COL) sub_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_COL) mul_bcast_cols_init_short(a, b);
    if constexpr (OP == MUL_ROW) mul_bcast_rows_init_short(a, b);
    if constexpr (OP == MUL) mul_tiles_init(a, b);
    if constexpr (OP == SUB) sub_tiles_init(a, b);
    cb_reserve_back(out, Wt);
    for (uint32_t i = 0; i < Wt; ++i) {
        tile_regs_acquire();
        if constexpr (OP == SUB_COL) sub_tiles_bcast_cols(a, b, i, 0, 0);
        if constexpr (OP == MUL_COL) mul_tiles_bcast_cols(a, b, i, 0, 0);
        if constexpr (OP == MUL_ROW) mul_tiles_bcast_rows(a, b, i, i, 0);
        if constexpr (OP == MUL) mul_tiles(a, b, i, i, 0);
        if constexpr (OP == SUB) sub_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out);
        tile_regs_release();
    }
    cb_push_back(out, Wt);
}

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    binary_op_init_common(cb_x, cb_scaler, cb_mean);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    if constexpr (DO_GAMMA) cb_wait_front(cb_gamma, Wt);
    for (uint32_t r = 0; r < num_rows; ++r) {
        row_mean(cb_x, cb_mean, false);                       // mean
        cb_wait_front(cb_mean, 1);
        stage<SUB_COL>(cb_x, cb_mean, cb_xc);                 // xc
        cb_pop_front(cb_x, Wt);
        cb_pop_front(cb_mean, 1);
        cb_wait_front(cb_xc, Wt);
        stage<MUL>(cb_xc, cb_xc, cb_tmp);                     // xc^2
        row_mean(cb_tmp, cb_var, true);                       // var
        // rstd = rsqrt(var + eps)
        cb_wait_front(cb_var, 1);
        reconfig_data_format(cb_var, cb_eps);
        pack_reconfig_data_format(cb_rstd);
        add_tiles_init(cb_var, cb_eps);
        cb_reserve_back(cb_rstd, 1);
        tile_regs_acquire();
        add_tiles(cb_var, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_rstd);
        tile_regs_release();
        cb_push_back(cb_rstd, 1);
        cb_pop_front(cb_var, 1);
        cb_wait_front(cb_rstd, 1);
        stage<MUL_COL>(cb_xc, cb_rstd, cb_norm);              // norm
        cb_pop_front(cb_xc, Wt);
        cb_wait_front(cb_g, Wt);
        if constexpr (DO_GAMMA) {
            stage<MUL_ROW>(cb_g, cb_gamma, cb_dn);            // dn = g * gamma
            cb_pop_front(cb_g, Wt);
        }
        constexpr uint32_t dn = DO_GAMMA ? cb_dn : cb_g;
        row_mean(dn, cb_a, false);                            // a = mean(dn)
        cb_wait_front(cb_norm, Wt);
        stage<MUL>(dn, cb_norm, cb_tmp);                      // dn * norm
        row_mean(cb_tmp, cb_b, true);                         // b = mean(dn * norm)
        cb_wait_front(cb_a, 1);
        stage<SUB_COL>(dn, cb_a, cb_t);                       // t = dn - a
        cb_pop_front(dn, Wt);
        cb_pop_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        stage<MUL_COL>(cb_norm, cb_b, cb_tmp);                // norm * b
        cb_pop_front(cb_norm, Wt);
        cb_pop_front(cb_b, 1);
        cb_wait_front(cb_t, Wt);
        cb_wait_front(cb_tmp, Wt);
        stage<SUB>(cb_t, cb_tmp, cb_xc);                      // v = t - norm * b
        cb_pop_front(cb_t, Wt);
        cb_pop_front(cb_tmp, Wt);
        cb_wait_front(cb_xc, Wt);
        stage<MUL_COL>(cb_xc, cb_rstd, cb_out);               // dx = v * rstd
        cb_pop_front(cb_xc, Wt);
        cb_pop_front(cb_rstd, 1);
    }
}
