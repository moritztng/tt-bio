// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A core's units in a program that splits output tiles over the cores. By default the per-core runtime args are
// (first, count), a contiguous range; under UNIT_LIST they are (count, u_0, ..., u_{count-1}), a list the host deals
// out (tt_bio/dit_pair_bias.py::_deal), so a caller can recompute only the tiles whose inputs changed.
#pragma once

struct CoreUnits {
    uint32_t n, u0;
    uint32_t operator[](uint32_t k) const {
#ifdef UNIT_LIST
        return get_arg_val<uint32_t>(1 + k);
#else
        return u0 + k;
#endif
    }
};

inline CoreUnits core_units() {
#ifdef UNIT_LIST
    return {get_arg_val<uint32_t>(0), 0};
#else
    return {get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(0)};
#endif
}
