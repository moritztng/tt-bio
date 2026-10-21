// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The gates' SFPU sigmoid, accurate or fast (tt_bio/gate_sigmoid.py, TT_BIO_SIGMOID_FAST).
// Accurate: metal's exp + reciprocal (sigmoid_tile). Fast: metal's SFPLUT piecewise-linear sigmoid
// (calculate_sigmoid_appx), about a relu's cost. It is called directly rather than through approx mode because
// under the silu_f32 overlay approx mode on an fp32 dest is calculate_silu_f32's sigmoid, and the fast mode has
// to be one function at every site. Its coefficients live in LREG0-2, which other SFPU ops overwrite, so
// gate_sigmoid_lut_tile reloads them per tile.
//
// Kernels without a SIGPOLY argument switch on the TT_BIO_SIG_FAST define (gate_sigmoid_init / _tile); the
// SIGPOLY kernels take SIGPOLY 3 (sigmoid_poly.hpp).
#pragma once

#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#endif

ALWI void gate_sigmoid_lut_tile(uint32_t idst) {
    MATH((ckernel::sfpu::sigmoid_appx_init()));
    MATH((_llk_math_eltwise_unary_sfpu_params_<true>(ckernel::sfpu::calculate_sigmoid_appx<8>, idst,
                                                     (int)VectorMode::RC)));
}

#ifdef TT_BIO_SIG_FAST
ALWI void gate_sigmoid_init() { sigmoid_tile_init(); }
ALWI void gate_sigmoid_tile(uint32_t idst) { gate_sigmoid_lut_tile(idst); }
#else
ALWI void gate_sigmoid_init() { sigmoid_tile_init(); }
ALWI void gate_sigmoid_tile(uint32_t idst) { sigmoid_tile(idst); }
#endif
