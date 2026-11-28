// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_mix compute: AF2's column attention over two MSA rows, from the projected q, k, v to the
// attended value, for one head of one token tile row (a head is 32 channels, one tile):
//
//   kd = k0 - k1                      vd = v0 - v1
//   gap_s = scale * rowsum(q_s * kd) + bias          (s = 0, 1; one number a token and head)
//   out_s = v1 + sigmoid(gap_s) * vd
//
// With two keys the softmax is a sigmoid of the logit gap, so nothing here leaves the tile. The row
// sum is a matmul with a constant tile full of `scale` (const tile 0), which also spreads it over
// the head's 32 columns; the bias column is spread the same way by a tile whose first row is ones
// (const tile 1). Everything else is the SFPU in a float32 DST, rounded once at the pack. The
// products q_s * kd go through a float32 CB because the FPU reads its operands from CBs.
//
// c_0 holds a unit's k0, k1, q0, q1, v0, v1 in that order.
//
// BW is the VJP, from the cotangent d of out (c_3: d0, d1) to that of q, k, v. With p = sigmoid(gap):
//
//   dv0 = p0 d0 + p1 d1                 dv1 = d0 + d1 - dv0
//   g_s = scale * rowsum(d_s * vd) * p_s (1 - p_s)
//   dq_s = g_s * kd                     dk0 = g_0 q0 + g_1 q1 = -dk1
//
// It recomputes kd, vd and p from q, k, v, so the tape holds the projection and nothing else. Both
// row sums are matmuls with const tile 0, so d_s * vd rides the float32 CB beside q_s * kd; p, p (1 - p)
// and g wait in float32 CBs of their own between acquires.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#include "unit.h"

namespace {
constexpr uint32_t cb_in = 0, cb_bias = 1, cb_const = 2, cb_d = 3, cb_out = 16;
constexpr uint32_t cb_mid = 24, cb_p = 25, cb_g = 26;   // float32 scratch between DST acquires
constexpr uint32_t BIAS = get_compile_time_arg_val(0);
constexpr uint32_t BW = get_compile_time_arg_val(1);

// A float32 DST gives the FPU's copy and the packer four tile slots, not eight: index 4 is slot 0 again
// (measured: a copy_tile to slot 4 came back from pack_tile(0)). So every acquire here stays in slots 0..3 and
// what has to outlive one goes through a float32 CB.

inline void copy_in(uint32_t cb, uint32_t idx, uint32_t slot) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, idx, slot);
}

// dst 0.. = scale * rowsum(mid tile first..) spread over the columns, plus the bias on the first two if asked
inline void head_sums(uint32_t first, uint32_t count, bool bias) {
    reconfig_data_format(cb_const, cb_mid);   // srcA takes in1 (the constant), srcB in0
    mm_init_short(cb_mid, cb_const, 0);
    for (uint32_t i = 0; i < count; ++i) {
        matmul_tiles(cb_mid, cb_const, first + i, 0, i);
    }
    if (bias) {
        reconfig_data_format(cb_const, cb_bias);
        mm_init_short(cb_bias, cb_const, 0);
        matmul_tiles(cb_bias, cb_const, 0, 1, 0);
        matmul_tiles(cb_bias, cb_const, 0, 1, 1);
    }
}

inline void pack(uint32_t cb, uint32_t a, uint32_t b) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(a, cb);
    pack_tile(b, cb);
    tile_regs_release();
}

// mid += (in[a] - in[a + 1]) * {x0, x1}; x from `cb` tiles i, i + 1
inline void diff_times(uint32_t a, uint32_t cb, uint32_t i) {
    cb_reserve_back(cb_mid, 2);
    tile_regs_acquire();
    copy_in(cb_in, a, 0);
    copy_in(cb_in, a + 1, 1);
    copy_in(cb, i, 2);
    copy_in(cb, i + 1, 3);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    mul_binary_tile_init();
    mul_binary_tile(2, 0, 2);
    mul_binary_tile(3, 0, 3);
    pack(cb_mid, 2, 3);
    cb_push_back(cb_mid, 2);
}

inline void forward() {
    diff_times(0, cb_in, 2);                  // q_s * kd
    cb_wait_front(cb_mid, 2);
    cb_reserve_back(cb_out, 2);
    tile_regs_acquire();
    head_sums(0, 2, BIAS);
    copy_in(cb_in, 4, 2);
    copy_in(cb_in, 5, 3);
    sigmoid_tile_init();
    sigmoid_tile(0);
    sigmoid_tile(1);
    sub_binary_tile_init();
    sub_binary_tile(2, 3, 2);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    mul_binary_tile(1, 2, 1);
    add_binary_tile_init();
    add_binary_tile(0, 3, 0);
    add_binary_tile(1, 3, 1);
    pack(cb_out, 0, 1);
    cb_push_back(cb_out, 2);
    cb_pop_front(cb_mid, 2);
}

inline void backward() {
    diff_times(0, cb_in, 2);                  // mid 0, 1 = q_s * kd
    diff_times(4, cb_d, 0);                   // mid 2, 3 = d_s * vd
    cb_wait_front(cb_mid, 4);
    cb_reserve_back(cb_out, 6);

    // p_s and p_s (1 - p_s)
    cb_reserve_back(cb_p, 4);
    tile_regs_acquire();
    head_sums(0, 2, BIAS);
    sigmoid_tile_init();
    sigmoid_tile(0);
    sigmoid_tile(1);
    mul_binary_tile_init();
    mul_binary_tile(0, 0, 2);
    mul_binary_tile(1, 1, 3);
    sub_binary_tile_init();
    sub_binary_tile(0, 2, 2);
    sub_binary_tile(1, 3, 3);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_p);
    for (uint32_t i = 0; i < 4; ++i) {
        pack_tile(i, cb_p);
    }
    tile_regs_release();
    cb_push_back(cb_p, 4);
    cb_wait_front(cb_p, 4);

    // g_s, and dq_s = g_s * kd
    cb_reserve_back(cb_g, 2);
    tile_regs_acquire();
    head_sums(2, 2, false);
    copy_in(cb_p, 2, 2);
    copy_in(cb_p, 3, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    mul_binary_tile(1, 3, 1);
    copy_in(cb_in, 0, 2);
    copy_in(cb_in, 1, 3);
    sub_binary_tile_init();
    sub_binary_tile(2, 3, 2);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 3);
    mul_binary_tile(2, 1, 2);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(3, cb_out);
    pack_tile(2, cb_out);
    pack_reconfig_data_format(cb_g);
    pack_tile(0, cb_g);
    pack_tile(1, cb_g);
    tile_regs_release();
    cb_push_back(cb_g, 2);
    cb_wait_front(cb_g, 2);

    // dk0 = g_0 q0 + g_1 q1, dk1 = -dk0
    tile_regs_acquire();
    copy_in(cb_g, 0, 0);
    copy_in(cb_g, 1, 1);
    copy_in(cb_in, 2, 2);
    copy_in(cb_in, 3, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    mul_binary_tile(1, 3, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    sub_binary_tile_init();
    sub_binary_tile(1, 1, 1);
    sub_binary_tile(1, 0, 1);
    pack(cb_out, 0, 1);

    // dv0 = p0 d0 + p1 d1, dv1 = d0 + d1 - dv0
    tile_regs_acquire();
    copy_in(cb_p, 0, 0);
    copy_in(cb_p, 1, 1);
    copy_in(cb_d, 0, 2);
    copy_in(cb_d, 1, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    mul_binary_tile(1, 3, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(2, 3, 2);
    sub_binary_tile_init();
    sub_binary_tile(2, 0, 2);
    pack(cb_out, 0, 2);

    cb_push_back(cb_out, 6);
    cb_pop_front(cb_mid, 4);
    cb_pop_front(cb_p, 4);
    cb_pop_front(cb_g, 2);
}
}  // namespace

void kernel_main() {
    uint32_t num_units;
    {
        constexpr uint32_t G = 2;
        constexpr uint32_t COMPACT = get_compile_time_arg_val(G);
        if constexpr (COMPACT) {
            num_units = genq::slice<get_compile_time_arg_val(G + 1), get_compile_time_arg_val(G + 2),
                                    get_compile_time_arg_val(G + 3), get_compile_time_arg_val(G + 4),
                                    get_compile_time_arg_val(G + 5), get_compile_time_arg_val(G + 6),
                                    get_compile_time_arg_val(G + 7)>(
                            get_absolute_logical_x(), get_absolute_logical_y()).num;
        } else {
            num_units = get_arg_val<uint32_t>(0);
        }
    }

    binary_op_init_common(cb_in, cb_const, cb_out);
    cb_wait_front(cb_const, 2);
    for (uint32_t u = 0; u < num_units; ++u) {
        cb_wait_front(cb_in, 6);
        if constexpr (BIAS) {
            cb_wait_front(cb_bias, 1);
        }

        if constexpr (BW) {
            cb_wait_front(cb_d, 2);
            backward();
        } else {
            forward();
        }
        cb_pop_front(cb_in, 6);
        if constexpr (BW) {
            cb_pop_front(cb_d, 2);
        }
        if constexpr (BIAS) {
            cb_pop_front(cb_bias, 1);
        }
    }
}
