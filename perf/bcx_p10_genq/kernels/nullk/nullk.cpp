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
    volatile uint32_t acc = get_common_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < NWORDS; ++i) {
        acc += get_arg_val<uint32_t>(i);
    }
    (void)acc;
}
