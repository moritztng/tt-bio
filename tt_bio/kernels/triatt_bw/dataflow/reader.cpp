// SPDX-License-Identifier: Apache-2.0
//
// Reader for the triangle-attention backward. Per query chunk it fronts that chunk's rows of the
// bias, then streams every leading-axis row of this core's group: the chunk's q and dO tiles and
// the whole of k and v. With more than one chunk, dK and dV sum across chunks, so for every chunk
// after the first it also reads back the float32 partial the writer left for the same row, and
// it waits on this core's semaphore until the writer says that write has landed.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr uint32_t qkv_tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t bias_tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t Qt = get_compile_time_arg_val(5);
    constexpr uint32_t acc_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);

    constexpr auto q_args = TensorAccessorArgs<8>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto do_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<do_args.next_compile_time_args_offset()>();
    constexpr auto dk_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto dv_args = TensorAccessorArgs<dk_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t do_addr = get_arg_val<uint32_t>(3);
    const uint32_t bias_addr = get_arg_val<uint32_t>(4);
    const uint32_t head = get_arg_val<uint32_t>(5);
    const uint32_t row_start = get_arg_val<uint32_t>(6);
    const uint32_t row_end = get_arg_val<uint32_t>(7);
    const uint32_t dk_addr = get_arg_val<uint32_t>(8);
    const uint32_t dv_addr = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_do = tt::CBIndex::c_3;
    constexpr uint32_t cb_bias = tt::CBIndex::c_4;
    constexpr uint32_t cb_prev = tt::CBIndex::c_9;

    constexpr uint32_t chunks = Nt / Qt;
    constexpr uint32_t col_tiles = Nt * Dt;
    constexpr uint32_t qcol_tiles = Qt * Dt;
    constexpr uint32_t chunk_bias_tiles = Qt * Nt;

    const auto q_reader = TensorAccessor(q_args, q_addr, qkv_tile_bytes);
    const auto k_reader = TensorAccessor(k_args, k_addr, qkv_tile_bytes);
    const auto v_reader = TensorAccessor(v_args, v_addr, qkv_tile_bytes);
    const auto do_reader = TensorAccessor(do_args, do_addr, qkv_tile_bytes);
    const auto bias_reader = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);
    const auto dk_reader = TensorAccessor(dk_args, dk_addr, acc_tile_bytes);
    const auto dv_reader = TensorAccessor(dv_args, dv_addr, acc_tile_bytes);

    volatile tt_l1_ptr uint32_t* written =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    const uint32_t rows = row_end - row_start;

    for (uint32_t c = 0; c < chunks; ++c) {
        {
            cb_reserve_back(cb_bias, chunk_bias_tiles);
            uint32_t ptr = get_write_ptr(cb_bias);
            const uint32_t base = head * Nt * Nt + c * chunk_bias_tiles;
            for (uint32_t i = 0; i < chunk_bias_tiles; ++i) {
                noc_async_read_tile(base + i, bias_reader, ptr);
                ptr += bias_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_bias, chunk_bias_tiles);
        }

        for (uint32_t row = row_start; row < row_end; ++row) {
            const uint32_t base = (row * H + head) * col_tiles;
            const uint32_t qbase = base + c * qcol_tiles;

            cb_reserve_back(cb_q, qcol_tiles);
            cb_reserve_back(cb_k, col_tiles);
            cb_reserve_back(cb_v, col_tiles);
            cb_reserve_back(cb_do, qcol_tiles);
            uint32_t qp = get_write_ptr(cb_q);
            uint32_t kp = get_write_ptr(cb_k);
            uint32_t vp = get_write_ptr(cb_v);
            uint32_t dp = get_write_ptr(cb_do);
            for (uint32_t i = 0; i < qcol_tiles; ++i) {
                noc_async_read_tile(qbase + i, q_reader, qp);
                noc_async_read_tile(qbase + i, do_reader, dp);
                qp += qkv_tile_bytes;
                dp += qkv_tile_bytes;
            }
            for (uint32_t i = 0; i < col_tiles; ++i) {
                noc_async_read_tile(base + i, k_reader, kp);
                noc_async_read_tile(base + i, v_reader, vp);
                kp += qkv_tile_bytes;
                vp += qkv_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_q, qcol_tiles);
            cb_push_back(cb_k, col_tiles);
            cb_push_back(cb_v, col_tiles);
            cb_push_back(cb_do, qcol_tiles);

            if (c > 0) {
                // The writer bumps the semaphore once per (chunk, row) after its write barrier,
                // so this row's partial from chunk c-1 is in DRAM once the count passes it.
                noc_semaphore_wait_min(written, (c - 1) * rows + (row - row_start) + 1);
                // dV first, then dK: the order the compute kernel adds them in.
                for (uint32_t which = 0; which < 2; ++which) {
                    const auto& src = which == 0 ? dv_reader : dk_reader;
                    cb_reserve_back(cb_prev, col_tiles);
                    uint32_t pp = get_write_ptr(cb_prev);
                    for (uint32_t i = 0; i < col_tiles; ++i) {
                        noc_async_read_tile(base + i, src, pp);
                        pp += acc_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_prev, col_tiles);
                }
            }
        }
    }
}
