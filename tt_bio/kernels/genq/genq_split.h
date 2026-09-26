// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Recompute this core's slice of a `split_work_to_cores` split from its own logical coordinates.
//
// A `generic_op` dispatch is charged per (core, kernel) that carries PER-CORE runtime arguments
// and is flat in how many words each one carries -- 130 cores x 3 kernels costs 0.0230 ms with
// two words a core and 0.0079 ms with none (`perf/bcx_p10_genq/nullk.py`). So a kernel that can
// derive its own `(first_unit, num_units)` costs a third of what the same kernel costs when the
// host hands it the same two numbers.
//
// The seven constants come from `tt_bio/genq.py::compact_plan`, which only returns them after
// checking that this arithmetic reproduces the host's own placement loop for every placed core.
// Nothing here is inferred on the device.
#pragma once

#include <cstdint>

namespace genq {

constexpr uint32_t ROW_MAJOR = 0;

struct Slice {
    uint32_t first;
    uint32_t num;
};

// ORDER/X0/Y0/SPAN place this core in the scan the host used; N1/P1/P2 are the split itself:
// the first N1 cores take P1 units each and the rest take P2.
template <uint32_t ORDER, uint32_t X0, uint32_t Y0, uint32_t SPAN,
          uint32_t N1, uint32_t P1, uint32_t P2>
inline Slice slice(uint32_t x, uint32_t y) {
    const uint32_t i = (ORDER == ROW_MAJOR) ? (y - Y0) * SPAN + (x - X0)
                                            : (x - X0) * SPAN + (y - Y0);
    return (i < N1) ? Slice{i * P1, P1} : Slice{N1 * P1 + (i - N1) * P2, P2};
}

}  // namespace genq
