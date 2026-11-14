// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triatt_ln compute (tt_bio/triatt_ln.py): per row tile of the raw pair x, the channel LayerNorm
// (kernels/common/ln_prologue.hpp, as trimul T1) packed bf16 into c_1, then xn @ W for the NW
// output tiles [q | k | v | g | bias], SBW tiles a DST block, packed in n order to c_2.
// fp32 DEST (the statistics), the caller's fidelity. LNS: the two cores of a column-group pair each norm every
// other row and project the partner's xn for the rest, so neither norms a row twice.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/transpose_wh.h"
#include "../trimul_gin_moved/sigmoid_poly.hpp"
#include "../common/ln_prologue.hpp"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t NW = get_compile_time_arg_val(1);
constexpr uint32_t RNE = get_compile_time_arg_val(2);   // 1: round to nearest even before the pack
// Bench-only stage skips (triatt_ln.ABL, perf/spd_m_protenix/tl_attr.py), the output then wrong by design: 1 the
// layer_norm, 2 the projection, 4 the writer's DRAM writes, 8 the reader's DRAM reads of x. 0 in production.
constexpr uint32_t ABL = get_compile_time_arg_val(3);
#ifndef NQ
#define NQ 0   // triatt_ln_b8: tiles n < nq (runtime arg 1: q, k, v) pack bfp8 to c_2, the rest bf16 to c_4 at n - nq
#endif
#ifndef HM
#define HM 0   // triatt_ln.BIAS_HEADS: the bias tile (this group's tile runtime arg 3) leaves transposed, h rows x j
#endif
#ifndef LNS
#define LNS 0  // triatt_ln.LN_SHARE: this core norms rows i % 2 == runtime arg 2 and takes the others' xn in c_5
#endif
constexpr uint32_t SBW = 4;                              // fp32 half-sync DST holds 4 tiles
constexpr uint32_t NFULL = NW / SBW * SBW;

constexpr uint32_t w_cb = tt::CBIndex::c_0;      // W [Kt x NW], resident
constexpr uint32_t xn_cb = tt::CBIndex::c_1;     // layer_norm(x) of one row tile, Kt tiles
constexpr uint32_t xp_cb = tt::CBIndex::c_5;     // LNS: the partner core's xn of the rows it norms (writer.cpp)
constexpr uint32_t tok_cb = tt::CBIndex::c_6;    // LNS: one empty page per own xn, tells the writer it is in c_1
constexpr uint32_t out_cb = tt::CBIndex::c_2;    // NW tiles a row tile to the writer
constexpr uint32_t cb_x = tt::CBIndex::c_7;
constexpr uint32_t cb_gamma = tt::CBIndex::c_8;
constexpr uint32_t cb_beta = tt::CBIndex::c_9;
constexpr uint32_t cb_scaler = tt::CBIndex::c_10;
constexpr uint32_t cb_eps = tt::CBIndex::c_11;
constexpr uint32_t cb_mean = tt::CBIndex::c_12;
constexpr uint32_t cb_var = tt::CBIndex::c_13;
constexpr uint32_t cb_rstd = tt::CBIndex::c_14;
constexpr uint32_t cb_xc = tt::CBIndex::c_16;
constexpr uint32_t cb_tmp = tt::CBIndex::c_17;
constexpr uint32_t bias_cb = tt::CBIndex::c_18;  // HM: the bias tile packed as it is, unpacked transposed

struct LnCbs {
    static constexpr uint32_t Kt = ::Kt, x = cb_x, gamma = cb_gamma, beta = cb_beta, scaler = cb_scaler,
        eps = cb_eps, mean = cb_mean, var = cb_var, rstd = cb_rstd, xc = cb_xc, tmp = cb_tmp;
};

// HM: the bias tile is packed bf16 to c_18 and comes back through the unpacker's transpose, so it leaves exactly
// as the permute would have left it. transpose_wh_dest in DST after the matmul moved q, k, v and g as well as the
// bias (perf/mol1_triattn/bh_diag.py): it stalls on a srcA valid the matmul does not leave and clears the next one.
ALWI void transpose_bias(uint32_t a_cb, uint32_t cb, uint32_t slot) {
    cb_wait_front(bias_cb, 1);
    transpose_wh_init_short(bias_cb);
    tile_regs_acquire();
    transpose_wh_tile(bias_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb, slot);
    tile_regs_release();
    cb_pop_front(bias_cb, 1);
    mm_block_init_short(a_cb, w_cb, false, SBW, 1, Kt);   // the next full block's; a tail inits its own
    reconfig_data_format(w_cb, a_cb);
}

template <uint32_t W>
ALWI void project(uint32_t a_cb, uint32_t n0, uint32_t nq, uint32_t bi) {
    const uint32_t b = HM ? bi - n0 : W;   // the bias tile's place in this block, W (or more) if not here
    if (HM && b < W) cb_reserve_back(bias_cb, 1);
    tile_regs_acquire();
    uint32_t i1 = n0;
    for (uint32_t k = 0; k < Kt; ++k, i1 += NW) matmul_block(a_cb, w_cb, k, i1, 0, false, W, 1, Kt);
    if constexpr (RNE) {
        for (uint32_t i = 0; i < W; ++i) round_bf16_rne_tile(i);
    }
    tile_regs_commit();
    tile_regs_wait();
    // nq is a multiple of SBW, so a block never straddles the two buffers
    const uint32_t cb = NQ && n0 >= nq ? (uint32_t)tt::CBIndex::c_4 : out_cb, at = NQ && n0 >= nq ? n0 - nq : n0;
    if constexpr (NQ) pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < W; ++i) {
        if (HM && i == b) pack_tile<true>(i, bias_cb, 0);
        else pack_tile<true>(i, cb, at + i);
    }
    tile_regs_release();
    if (HM && b < W) {
        cb_push_back(bias_cb, 1);
        transpose_bias(a_cb, cb, at + b);
    }
}

void kernel_main() {
    const uint32_t nrows = get_arg_val<uint32_t>(0);
    const uint32_t nq = NQ ? get_arg_val<uint32_t>(1) : 0;
    const uint32_t bi = HM ? get_arg_val<uint32_t>(3) : 0xFFFF;

    mm_init(xn_cb, w_cb, out_cb, 0);
    cb_wait_front(w_cb, Kt * NW);
    cb_wait_front(cb_gamma, Kt);
    cb_wait_front(cb_beta, Kt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    // LNS walks rows in pairs (2p, 2p + 1): norm this core's row of the pair first, then project both, so the two
    // cores of a pair norm at the same time instead of each waiting on the other's norm between projections.
    const uint32_t own = LNS ? get_arg_val<uint32_t>(2) : 0;
    for (uint32_t r = 0; r < nrows; ++r) {
        const uint32_t o = LNS ? (r & ~1u) + own : r;   // the row this core norms in r's pair
        if (r == (LNS ? (r & ~1u) : r) && o < nrows) {
            cb_reserve_back(xn_cb, Kt);
            if constexpr (ABL & 1) {
                cb_wait_front(cb_x, Kt);
                cb_pop_front(cb_x, Kt);
            } else {
                ln_prologue::layer_norm_row<LnCbs>(xn_cb, [](uint32_t i) { pack_tile<true>(0, xn_cb, i); });
            }
            cb_push_back(xn_cb, Kt);
            if constexpr (LNS) {
                cb_reserve_back(tok_cb, 1);
                cb_push_back(tok_cb, 1);
            }
        }
        const uint32_t a_cb = r == o ? xn_cb : xp_cb;
        cb_wait_front(a_cb, Kt);

        cb_reserve_back(out_cb, NQ ? nq : NW);
        if constexpr (NQ) cb_reserve_back(tt::CBIndex::c_4, NW - nq);
        if constexpr (!(ABL & 2)) {
            mm_block_init_short(a_cb, w_cb, false, SBW, 1, Kt);
            reconfig_data_format(w_cb, a_cb);
            pack_reconfig_data_format(out_cb);
            for (uint32_t n0 = 0; n0 < NFULL; n0 += SBW) project<SBW>(a_cb, n0, nq, bi);
            if constexpr (NW % SBW) {
                mm_block_init_short(a_cb, w_cb, false, NW % SBW, 1, Kt);
                project<NW % SBW>(a_cb, NFULL, nq, bi);
            }
        }
        cb_push_back(out_cb, NQ ? nq : NW);
        if constexpr (NQ) cb_push_back(tt::CBIndex::c_4, NW - nq);
        cb_pop_front(a_cb, Kt);
    }
}
