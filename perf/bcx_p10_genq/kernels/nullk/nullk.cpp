// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A kernel that does nothing, so a dispatch can be priced with the device work removed.
//
// It touches no CB and issues no NOC transaction; it only reads its own arguments so the
// compiler cannot delete them. What varies between the descriptors that drive it is the number
// of CORES it is placed on and the number of per-core runtime-arg WORDS each one gets, which is
// exactly the axis `probe.py` needs to separate "the dispatch builds a program" from "the
// dispatch writes runtime args core by core".
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NWORDS = get_compile_time_arg_val(0);
    constexpr uint32_t NCOMMON = get_compile_time_arg_val(1);
    volatile uint32_t acc = 0;
    for (uint32_t i = 0; i < NWORDS; ++i) {
        acc += get_arg_val<uint32_t>(i);
    }
    // Touch the common table the way the compact path will: index it by this core's own
    // absolute logical coordinates rather than by a per-core argument.
    if (NCOMMON) {
        const uint32_t idx = get_absolute_logical_y() * 16u + get_absolute_logical_x();
        acc += get_common_arg_val<uint32_t>(idx < NCOMMON ? idx : 0);
    }
    (void)acc;
}
