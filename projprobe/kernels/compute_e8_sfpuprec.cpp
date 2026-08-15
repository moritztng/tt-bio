// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// E8f -- the SFPU's accuracy on the same two ops the FPU just failed.
//
// E8e measured an fp32 mul_tiles at 2.05e-3 relative and an add_tiles at 6.99e-4 at every fidelity,
// which is the FPU truncating its SrcA/SrcB operands to about 11 mantissa bits. §6's gate is 1e-5.
// The SFPU is a different unit: it operates on DST, and copy_tile with unpack-to-dest brings fp32
// across without going through SrcA at all. If mul_binary_tile and add_binary_tile come back at fp32
// accuracy, the coarse kernel's numeric path has to live on the SFPU and the whole §4.3 op budget
// re-prices at the SFPU's 4.7x. If they do not, this seam has no exact-trilinear kernel on this
// silicon at all.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_COL)

#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t op = get_compile_time_arg_val(2);

    init_sfpu(in_cb, out_cb);
    cb_wait_front(in_cb, 1);
    cb_reserve_back(out_cb, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short(in_cb);
    copy_tile(in_cb, 0, 0);
    copy_tile(in_cb, 0, 1);
    if constexpr (op == 0) {
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
    } else if constexpr (op == 1) {
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    } else if constexpr (op == 3) {
        unary_bcast_init<BroadcastType::ROW>(in_cb, out_cb);
        unary_bcast<BroadcastType::ROW>(in_cb, 0, 0);
    } else if constexpr (op == 5 || op == 6) {
        // The pixel sum in §4.3's compare stage is a reduce. reduce_tile takes an
        // enforce_fp32_accumulation template flag and E8h priced it at 61.60 ns, the same as the
        // ordinary one, so it is free if it is exact. 5 = enforced, 6 = not, so the flag's effect is
        // visible rather than assumed.
        constexpr bool enf = (op == 5);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, enf>(in_cb, in_cb, out_cb);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, enf>(in_cb, in_cb, 0, 0, 0);
        reduce_uninit<enf>();
    } else if constexpr (op == 4) {
        unary_bcast_init<BroadcastType::COL>(in_cb, out_cb);
        unary_bcast<BroadcastType::COL>(in_cb, 0, 0);
    } else {
        // The pass-through control: copy_tile alone, so a residual here would mean the fp32 never
        // reached DST intact and neither SFPU number could be trusted.
        ;
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);
    cb_pop_front(in_cb, 1);
}
