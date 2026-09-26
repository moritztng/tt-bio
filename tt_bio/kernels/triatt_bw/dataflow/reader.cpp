// SPDX-License-Identifier: Apache-2.0
//
// Reader for the triangle-attention backward.
//
// Two jobs. The head's bias slice is read ONCE and left fronted for the whole group: the bias is
// [1, H, N, N] broadcast over the leading axis, so nothing about it depends on the row, and
// re-reading it per row is what made the stock forward's mask cost 2.65x. Then q, k, v and dO
// stream one leading-axis row at a time, which is all the compute kernel ever needs resident.
//
// k is read in its natural [Nt, Dt] tile order and used twice by the compute kernel: as K^T for
// the scores (the tile order of a single-tile-wide operand is already its own transpose, so only
// the faces move, which matmul_block's transpose flag does) and as K for dQ. Reading it once is
// the difference between 84.94 MB and 106.2 MB per call.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr uint32_t qkv_tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t bias_tile_bytes = get_compile_time_arg_val(4);

    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto do_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<do_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t do_addr = get_arg_val<uint32_t>(3);
    const uint32_t bias_addr = get_arg_val<uint32_t>(4);
    const uint32_t head = get_arg_val<uint32_t>(5);
    const uint32_t row_start = get_arg_val<uint32_t>(6);
    const uint32_t row_end = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_do = tt::CBIndex::c_3;
    constexpr uint32_t cb_bias = tt::CBIndex::c_4;

    constexpr uint32_t col_tiles = Nt * Dt;
    constexpr uint32_t score_tiles = Nt * Nt;

    const auto q_reader = TensorAccessor(q_args, q_addr, qkv_tile_bytes);
    const auto k_reader = TensorAccessor(k_args, k_addr, qkv_tile_bytes);
    const auto v_reader = TensorAccessor(v_args, v_addr, qkv_tile_bytes);
    const auto do_reader = TensorAccessor(do_args, do_addr, qkv_tile_bytes);
    const auto bias_reader = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);

    // The head's whole bias grid, once, and never popped. [1, H, N, N] means the leading axis is
    // absent from the index, so this is the only read of it this core ever makes.
    {
        cb_reserve_back(cb_bias, score_tiles);
        uint32_t ptr = get_write_ptr(cb_bias);
        const uint32_t base = head * score_tiles;
        for (uint32_t i = 0; i < score_tiles; ++i) {
            noc_async_read_tile(base + i, bias_reader, ptr);
            ptr += bias_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_bias, score_tiles);
    }

    for (uint32_t row = row_start; row < row_end; ++row) {
        const uint32_t base = (row * H + head) * col_tiles;

        cb_reserve_back(cb_q, col_tiles);
        cb_reserve_back(cb_k, col_tiles);
        cb_reserve_back(cb_v, col_tiles);
        cb_reserve_back(cb_do, col_tiles);
        uint32_t qp = get_write_ptr(cb_q);
        uint32_t kp = get_write_ptr(cb_k);
        uint32_t vp = get_write_ptr(cb_v);
        uint32_t dp = get_write_ptr(cb_do);
        for (uint32_t i = 0; i < col_tiles; ++i) {
            // DIAGNOSTIC: issue order reversed. q and k come back zero and v/dO do not; if the
            // zeros follow the position rather than the tensor, this is about the first reads
            // issued, not about q and k.
            noc_async_read_tile(base + i, q_reader, qp);
            noc_async_read_tile(base + i, k_reader, kp);
            noc_async_read_tile(base + i, v_reader, vp);
            noc_async_read_tile(base + i, do_reader, dp);
            qp += qkv_tile_bytes;
            kp += qkv_tile_bytes;
            vp += qkv_tile_bytes;
            dp += qkv_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, col_tiles);
        cb_push_back(cb_k, col_tiles);
        cb_push_back(cb_v, col_tiles);
        cb_push_back(cb_do, col_tiles);
    }
}
