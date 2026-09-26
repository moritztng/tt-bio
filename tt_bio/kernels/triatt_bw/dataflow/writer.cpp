// SPDX-License-Identifier: Apache-2.0
//
// Writer for the triangle-attention backward.
//
// Drains dq, dk and dv one leading-axis row at a time, then writes this core's dbias partial.
//
// The partial is read straight out of the compute kernel's accumulator L1 rather than through a
// second circular buffer, which is why cb_done exists: a [Nt, Nt] float32 copy would have cost
// 331.8 KB of L1, and at the shipped 288-token shape that is the difference between the whole
// program fitting in L1 and not. cb_done carries no data, only the fact that the accumulator is
// final.
//
// This kernel also seeds the three constant tiles the compute kernel needs, the way the stock SDPA
// writer seeds its identity scalars: the reduction scaler, the attention scale as a broadcast
// scalar, and one all-zero tile that the accumulator is seeded from.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr uint32_t qkv_tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t partial_tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(5);
    constexpr uint32_t scale_scalar_packed = get_compile_time_arg_val(6);

    constexpr auto dq_args = TensorAccessorArgs<7>();
    constexpr auto dk_args = TensorAccessorArgs<dq_args.next_compile_time_args_offset()>();
    constexpr auto dv_args = TensorAccessorArgs<dk_args.next_compile_time_args_offset()>();
    constexpr auto dbias_args = TensorAccessorArgs<dv_args.next_compile_time_args_offset()>();

    const uint32_t dq_addr = get_arg_val<uint32_t>(0);
    const uint32_t dk_addr = get_arg_val<uint32_t>(1);
    const uint32_t dv_addr = get_arg_val<uint32_t>(2);
    const uint32_t dbias_addr = get_arg_val<uint32_t>(3);
    const uint32_t head = get_arg_val<uint32_t>(4);
    const uint32_t group = get_arg_val<uint32_t>(5);
    const uint32_t row_start = get_arg_val<uint32_t>(6);
    const uint32_t row_end = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_scalar = tt::CBIndex::c_5;
    constexpr uint32_t cb_zero = tt::CBIndex::c_6;
    constexpr uint32_t cb_scale = tt::CBIndex::c_7;
    constexpr uint32_t cb_dbias = tt::CBIndex::c_29;
    constexpr uint32_t cb_done = tt::CBIndex::c_30;
    constexpr uint32_t cb_dq = tt::CBIndex::c_16;
    constexpr uint32_t cb_dk = tt::CBIndex::c_17;
    constexpr uint32_t cb_dv = tt::CBIndex::c_18;

    constexpr uint32_t col_tiles = Nt * Dt;
    constexpr uint32_t score_tiles = Nt * Nt;

    generate_reduce_scaler(cb_scalar, identity_scalar_packed);
    generate_bcast_scalar(cb_scale, scale_scalar_packed);
    {
        // A genuine all-zero tile. The bcast-scalar generators only set the positions a broadcast
        // reads, and the accumulator is seeded by copying this tile whole.
        cb_reserve_back(cb_zero, 1);
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_zero));
        for (uint32_t i = 0; i < (1024 * 2) / 4; ++i) {
            p[i] = 0;
        }
        cb_push_back(cb_zero, 1);
    }

    const auto dq_writer = TensorAccessor(dq_args, dq_addr, qkv_tile_bytes);
    const auto dk_writer = TensorAccessor(dk_args, dk_addr, qkv_tile_bytes);
    const auto dv_writer = TensorAccessor(dv_args, dv_addr, qkv_tile_bytes);
    const auto dbias_writer = TensorAccessor(dbias_args, dbias_addr, partial_tile_bytes);

    for (uint32_t row = row_start; row < row_end; ++row) {
        const uint32_t base = (row * H + head) * col_tiles;

        cb_wait_front(cb_dv, col_tiles);
        cb_wait_front(cb_dq, col_tiles);
        cb_wait_front(cb_dk, col_tiles);
        uint32_t qp = get_read_ptr(cb_dq);
        uint32_t kp = get_read_ptr(cb_dk);
        uint32_t vp = get_read_ptr(cb_dv);
        for (uint32_t i = 0; i < col_tiles; ++i) {
            noc_async_write_tile(base + i, dq_writer, qp);
            noc_async_write_tile(base + i, dk_writer, kp);
            noc_async_write_tile(base + i, dv_writer, vp);
            qp += qkv_tile_bytes;
            kp += qkv_tile_bytes;
            vp += qkv_tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_dq, col_tiles);
        cb_pop_front(cb_dk, col_tiles);
        cb_pop_front(cb_dv, col_tiles);
    }

    // One [H, N, N] float32 slab per group of the leading axis. ttnn.sum over dim 0 of the
    // assembled [groups, H, N, N] tensor is the only reduction the host does.
    cb_wait_front(cb_done, 1);
    {
        const uint32_t base = (group * H + head) * score_tiles;
        uint32_t ptr = get_read_ptr(cb_dbias);
        for (uint32_t i = 0; i < score_tiles; ++i) {
            noc_async_write_tile(base + i, dbias_writer, ptr);
            ptr += partial_tile_bytes;
        }
        noc_async_write_barrier();
    }
}
