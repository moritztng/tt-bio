// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rne_add compute: out = round_rne_bf16(a + b), a and b bfloat16, one pass.
//
// The whole point of this kernel is the WIDTH of the intermediate. `ttnn.add` on two bfloat16
// operands does not compute this function -- measured against a float64 reference it misses on
// 11.0 % of random pairs, 6.1 % at a 256x operand ratio and 50.1 % on real ties -- and a mixed
// add with one float32 operand misses on exactly the same elements, so the datapath follows the
// narrowest operand and `dtype=` cannot buy width (`state/perf10/bcx-CALLS.md`, leg 2). AF2 buys
// the wide answer with four ttnn calls and 30 B/element; this buys it with one and 6.
//
// Three things are load-bearing, and all three are GRADED against a float64 host reference in
// `perf/bcx_p10_rneker/grade.py` rather than argued. The measurements are on qb2 card 0, 4 cases
// x 82,944 elements, AICLK 1350 during:
//
//   * `fp32_dest_acc_en=True` in the descriptor, so DEST is 32 bits. Packing the float32 DEST
//     into a bfloat16 CB with `add_tiles` and nothing else reproduces `ttnn.add`'s answer element
//     for element (97,234 of 331,776 wrong, the same 97,234), for two reasons that cancel into
//     one number: see the next two points.
//
//   * ADD_MODE=1, the SFPU add. Read back as float32 the SFPU add is the exact sum on 82,944 of
//     82,944 elements; the FPU's `add_tiles` is exact on 76,488 of 82,944 (92.2 %) at the same
//     32-bit DEST. The FPU arm is kept because it is the cheaper instruction and it is the
//     control that shows the loss is in the adder and not in the pack.
//
//   * ROUND_MODE=1, `rne_bf16_in_place` below. The PACKER converts a float32 DEST to bfloat16 by
//     rounding ties AWAY FROM ZERO (`reblock_permute_gated`'s header records the same thing from
//     the other side), so with an exact sum in DEST it is still wrong on 50.055 % of real ties.
//     Rounding to even in the SFPU first leaves a value that is already exactly bfloat16-
//     representable, and the pack becomes a pure format change with no tie left to break.
//
// `widen_add` is the same program at ADD_MODE=1, ROUND_MODE=0 with a float32 output CB: the
// backward's gradient fan-in, `f32(a) + f32(b)` rounded once to float32. A bfloat16 operand widens
// exactly on the way into DEST; a float32 one is unpacked straight to DEST (`UnpackToDestFp32`,
// set by the host), because through SrcA it would lose 13 mantissa bits to the 19-bit register.
//
// WHAT DOES NOT WORK, so nobody spends the day on it again: `typecast_tile<Float32, Float16_b>`
// is the right arithmetic -- it is `bits + 0x7fff + lsb`, which IS round-half-to-even -- and it
// cannot be used from a 32-bit-DEST kernel. Its store is scheduled through SFPLOADMACRO, and the
// DEST word it leaves is a 32-bit store on one face and a 16-bit store on the others: face 0 of
// each tile reads back as the biased float32 and faces 1-3 read back as `0x0000xxxx`. Graded, it
// is wrong on 93.6-96.8 % of elements whether or not the low half is then masked off. The
// arithmetic below is that function's, open-coded, with a 32-bit store.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

#include "../genq/genq_split.h"

#ifdef TRISC_MATH
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Round a float32 DEST tile to bfloat16 precision, ties to even, in place and still float32.
//
// Add 0x7fff plus the result's own bfloat16 mantissa LSB, then clear everything below that
// mantissa. The bias makes a strict majority round up and a strict minority round down; the LSB
// term is what sends an exact midpoint to the even neighbour instead of always upwards. Sign is
// bit 31 and the magnitude is sign-and-magnitude below it, so the integer add is correct for
// negatives too, and the only carry that reaches the exponent is the intended overflow of the
// largest finite bfloat16 to infinity.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void rne_bf16_in_place() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt v = sfpi::dst_reg[0];
        sfpi::vInt lsb = (v >> 16) & 1;
        v = v + 0x7fff + lsb;
        sfpi::dst_reg[0] = v & static_cast<int>(0xFFFF0000);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
#endif

static_assert(DST_ACCUM_MODE == 1, "rne_add needs a 32-bit DEST (fp32_dest_acc_en)");

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);    // c_0
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);    // c_1
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);  // c_16
    constexpr uint32_t GRAN = get_compile_time_arg_val(3);
    constexpr uint32_t ADD_MODE = get_compile_time_arg_val(4);    // 0 = FPU add_tiles, 1 = SFPU
    constexpr uint32_t ROUND_MODE = get_compile_time_arg_val(5);  // 0 = packer, 1 = SFPU RNE

    // See the reader for what GENQ_COMPACT buys and what checks it. Only the tile COUNT is
    // needed here: this kernel never addresses a tensor, it drains what the reader pushes.
    constexpr uint32_t G = 6;
    constexpr uint32_t COMPACT = get_compile_time_arg_val(G);
    uint32_t num_tiles;
    if constexpr (COMPACT) {
        num_tiles = genq::slice<get_compile_time_arg_val(G + 1), get_compile_time_arg_val(G + 2),
                                get_compile_time_arg_val(G + 3), get_compile_time_arg_val(G + 4),
                                get_compile_time_arg_val(G + 5), get_compile_time_arg_val(G + 6),
                                get_compile_time_arg_val(G + 7)>(
            get_absolute_logical_x(), get_absolute_logical_y()).num;
    } else {
        num_tiles = get_arg_val<uint32_t>(0);
    }

    // The SFPU add holds both operands in DEST, so it costs two slots a tile where the FPU add
    // costs one. A 32-bit DEST is four tiles deep, which is where the host's GRAN cap comes from.
    constexpr uint32_t SLOTS = (ADD_MODE == 1) ? 2 : 1;

    binary_op_init_common(cb_a, cb_b, cb_out);

    for (uint32_t i = 0; i < num_tiles; i += GRAN) {
        const uint32_t n = (num_tiles - i < GRAN) ? (num_tiles - i) : GRAN;

        cb_wait_front(cb_a, n);
        cb_wait_front(cb_b, n);
        cb_reserve_back(cb_out, n);

        tile_regs_acquire();
        if constexpr (ADD_MODE == 1) {
            for (uint32_t j = 0; j < n; ++j) {
                // `_with_dt` reconfigures the unpacker only when the two CBs differ in format,
                // which is `widen_add`'s float32 accumulator beside a bfloat16 contribution.
                // For rne_add both are bfloat16 and the reconfig is a compare.
                copy_tile_to_dst_init_short_with_dt(cb_b, cb_a);
                copy_tile(cb_a, j, SLOTS * j);
                copy_tile_to_dst_init_short_with_dt(cb_a, cb_b);
                copy_tile(cb_b, j, SLOTS * j + 1);
                add_binary_tile_init();
                add_binary_tile(SLOTS * j, SLOTS * j + 1, SLOTS * j);
            }
        } else {
            add_tiles_init(cb_a, cb_b);
            for (uint32_t j = 0; j < n; ++j) {
                add_tiles(cb_a, cb_b, j, j, j);
            }
        }
        if constexpr (ROUND_MODE == 1) {
            // `SfpuType::bitwise_and` only selects the generic SFPU init -- there is no callback
            // and no parameter, and the functor below is the whole op.
            MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::bitwise_and, false>()));
            for (uint32_t j = 0; j < n; ++j) {
                MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
                    ckernel::sfpu::rne_bf16_in_place<false>, SLOTS * j,
                    static_cast<int>(VectorMode::RC))));
            }
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(SLOTS * j, cb_out);
        }
        tile_regs_release();

        cb_pop_front(cb_a, n);
        cb_pop_front(cb_b, n);
        cb_push_back(cb_out, n);
    }
}
