// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute half of the do-nothing kernel. Same contract as nullk.cpp: read the args, touch
// nothing. A program cannot place two kernels of one processor class on one core, so the
// three-kernel point needs one of each class. Entry point is `kernel_main`, as every compute
// kernel in this codebase spells it.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    constexpr uint32_t NWORDS = get_compile_time_arg_val(0);
    volatile uint32_t acc = 0;
    for (uint32_t i = 0; i < NWORDS; ++i) {
        acc += get_arg_val<uint32_t>(i);
    }
    (void)acc;
}
