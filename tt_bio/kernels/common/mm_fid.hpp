// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mm_block_init_short and matmul_block at a fidelity the kernel picks (FID) instead of the program's MATH_FIDELITY,
// so one GEMM can run at HiFi2 while the program's element-wise multiplies keep the compute config's fidelity.
// The calls api/compute/matmul.h makes, Blackhole's dynamic throttle included; FID == MATH_FIDELITY is that code.
#pragma once
#include "api/compute/matmul.h"

namespace mm_fid {

template <MathFidelity FID>
ALWI void block_init_short(uint32_t in0, uint32_t in1, uint32_t ct, uint32_t rt, uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0, ct, rt, kt)));
    MATH((llk_math_matmul_init<FID, MM_THROTTLE>(in0, in1, 0, ct, rt)));
#ifdef ARCH_BLACKHOLE
    MATH((throttled_mop_status = 0));
#endif
}

#ifdef ARCH_BLACKHOLE
template <MathFidelity FID>
ALWI void block_math(uint32_t in0, uint32_t in1, uint32_t idst, uint32_t ct, uint32_t rt) {
    volatile uint32_t en = *(throttle_ptr) % 2;
    if (en) {
        if (throttled_mop_status != 1) {
            llk_math_matmul_init<FID, MM_THROTTLE_MAX>(in0, in1, 0, ct, rt);
            throttled_mop_status = 1;
        }
        llk_math_matmul<FID, MM_THROTTLE_MAX>(idst, ct, rt);
    } else {
        if (throttled_mop_status != 0) {
            llk_math_matmul_init<FID, MM_THROTTLE>(in0, in1, 0, ct, rt);
            throttled_mop_status = 0;
        }
        llk_math_matmul<FID, MM_THROTTLE>(idst, ct, rt);
    }
}
#endif

template <MathFidelity FID>
ALWI void block(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst, uint32_t ct, uint32_t rt,
                uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct, rt, kt)));
#ifdef ARCH_BLACKHOLE
    MATH((block_math<FID>(in0, in1, idst, ct, rt)));
#else
    MATH((llk_math_matmul<FID, MM_THROTTLE>(idst, ct, rt)));
#endif
}

}  // namespace mm_fid
