// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gather_tiled compute: tilize each 32-row block of CT tiles. Data movement only, every value passes unchanged.
#include <cstdint>

#define REDUCE_OP PoolType::SUM          // tilize.h guards tilize_block on these
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm = 0, cb_out = 16;
    const uint32_t nunits = get_arg_val<uint32_t>(1);
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
}
