// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gather_tiled compute: tilize each 32-row block of CT tiles. Data movement only, every value passes unchanged.
// GT_FMA: the tilized block goes to an intermediate CB and comes out as a + block * inv (inv one tile, its first column
// broadcast across the columns): the product on the FPU into dest, a added with dest reused as srcB, one rounding to
// bf16 at the pack. GT_ROWLIST: runtime arg 1 is replaced by the core's share of the host list (common args n, ncores).
#include <cstdint>

#define REDUCE_OP PoolType::SUM          // tilize.h guards tilize_block on these
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_hw_startup.h"
#ifdef GT_FMA
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#endif

void kernel_main() {
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm = 0, cb_out = 16;
#ifdef GT_ROWLIST
    const uint32_t me = get_arg_val<uint32_t>(2);
    const uint32_t n = get_common_arg_val<uint32_t>(0), ncores = get_common_arg_val<uint32_t>(1);
    const uint32_t nunits = me < n ? (n - 1 - me) / ncores + 1 : 0;
#else
    const uint32_t nunits = get_arg_val<uint32_t>(1);
#endif
#ifdef GT_FMA
    constexpr uint32_t cb_a = 3, cb_inv = 4, cb_em = 24;
    constexpr uint32_t GRAN = 4;   // tiles per dest acquire: four fp32 tiles fill half of Blackhole's dest
    // compute_kernel_hw_startup first: tilize_init configures no unpacker/math/packer state (pzgchk4, 10-17).
    compute_kernel_hw_startup(cb_rm, cb_inv, cb_out);
    for (uint32_t u = 0; u < nunits; ++u) {
        tilize_init(cb_rm, CT, cb_em);
        cb_wait_front(cb_rm, CT);
        cb_reserve_back(cb_em, CT);
        tilize_block(cb_rm, CT, cb_em);
        cb_push_back(cb_em, CT);
        cb_pop_front(cb_rm, CT);
        tilize_uninit(cb_rm, cb_em);
        cb_wait_front(cb_em, CT);
        cb_wait_front(cb_a, CT);
        cb_wait_front(cb_inv, 1);
        cb_reserve_back(cb_out, CT);
        for (uint32_t t0 = 0; t0 < CT; t0 += GRAN) {
            const uint32_t g = CT - t0 < GRAN ? CT - t0 : GRAN;
            tile_regs_acquire();
            mul_bcast_cols_init_short(cb_em, cb_inv);
            for (uint32_t j = 0; j < g; ++j) mul_tiles_bcast_cols(cb_em, cb_inv, t0 + j, 0, j);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_a);
            for (uint32_t j = 0; j < g; ++j)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_a, t0 + j, j);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < g; ++j) pack_tile(j, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, CT);
        cb_pop_front(cb_em, CT);
        cb_pop_front(cb_a, CT);
        cb_pop_front(cb_inv, 1);
    }
#else
    // tilize_init configures no unpacker/math/packer hardware state (formats, pack sync, and on Wormhole no pack
    // init at all). Without this the kernel inherits whatever the previous program on these cores left, and some
    // rows came out wrong in-model while the same call alone was exact (pzgchk4, 10-17).
    compute_kernel_hw_startup(cb_rm, cb_out);
    tilize_init(cb_rm, CT, cb_out);
    for (uint32_t u = 0; u < nunits; ++u) {
        cb_wait_front(cb_rm, CT);
        cb_reserve_back(cb_out, CT);
        tilize_block(cb_rm, CT, cb_out);
        cb_push_back(cb_out, CT);
        cb_pop_front(cb_rm, CT);
    }
    tilize_uninit(cb_rm, cb_out);
#endif
}
