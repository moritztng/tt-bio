// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gather_tiled compute: tilize each 32-row block of CT tiles. Data movement only, every value passes unchanged.
// With GT_FUSE: out = tilize(rows) * scale (broadcast along columns) + shift, the same two roundings as
// ttnn.multiply then ttnn.add: the product is rounded to bf16 (RNE) in an fp32 DST before the add, the sum by the packer.
#include <cstdint>

#define REDUCE_OP PoolType::SUM          // tilize.h guards tilize_block on these
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_hw_startup.h"
#ifdef GT_FUSE
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "../trimul_gin_moved/sigmoid_poly.hpp"
#endif

void kernel_main() {
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm = 0, cb_out = 16;
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    // tilize_init configures no unpacker/math/packer hardware state (formats, pack sync, and on Wormhole no pack
    // init at all). Without this the kernel inherits whatever the previous program on these cores left, and some
    // rows came out wrong in-model while the same call alone was exact (pzgchk4, 10-17).
    compute_kernel_hw_startup(cb_rm, cb_out);
#ifndef GT_FUSE
    tilize_init(cb_rm, CT, cb_out);
    for (uint32_t u = 0; u < nunits; ++u) {
        cb_wait_front(cb_rm, CT);
        cb_reserve_back(cb_out, CT);
        tilize_block(cb_rm, CT, cb_out);
        cb_push_back(cb_out, CT);
        cb_pop_front(cb_rm, CT);
    }
    tilize_uninit(cb_rm, cb_out);
#else
    constexpr uint32_t cb_sc = 3, cb_sh = 4, cb_em = 24;
    for (uint32_t u = 0; u < nunits; ++u) {
        tilize_init(cb_rm, CT, cb_em);
        cb_wait_front(cb_rm, CT);
        cb_reserve_back(cb_em, CT);
        tilize_block(cb_rm, CT, cb_em);
        cb_push_back(cb_em, CT);
        cb_pop_front(cb_rm, CT);
        tilize_uninit(cb_rm, cb_em);

        cb_wait_front(cb_em, CT);
        cb_wait_front(cb_sc, 1);
        cb_wait_front(cb_sh, CT);
        cb_reserve_back(cb_out, CT);
        reconfig_data_format(cb_em, cb_sc);
        pack_reconfig_data_format(cb_out);
        for (uint32_t t = 0; t < CT; ++t) {
            tile_regs_acquire();
            mul_bcast_cols_init_short(cb_em, cb_sc);
            mul_tiles_bcast_cols(cb_em, cb_sc, t, 0, 0);
            round_bf16_rne_tile(0);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_sh);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_sh, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, CT);
        cb_pop_front(cb_em, CT);
        cb_pop_front(cb_sc, 1);
        cb_pop_front(cb_sh, CT);
    }
#endif
}
