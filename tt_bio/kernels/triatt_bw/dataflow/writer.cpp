// SPDX-License-Identifier: Apache-2.0
//
// Writer for the triangle-attention backward. Per query chunk: every row's dQ tiles for the
// chunk and its whole dK and dV (float32 partials when there is more than one chunk, each chunk
// overwriting the last with the running sum), then the chunk's rows of the dbias partial straight
// out of the compute kernel's accumulator once cb_done says it is final.

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
    constexpr uint32_t Qt = get_compile_time_arg_val(7);
    constexpr uint32_t kv_tile_bytes = get_compile_time_arg_val(8);   // dK/dV: bf16, or f32 partials
    constexpr uint32_t sem_id = get_compile_time_arg_val(9);

    constexpr auto dq_args = TensorAccessorArgs<10>();
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
    constexpr uint32_t cb_ones = tt::CBIndex::c_8;
    constexpr uint32_t cb_dbias = tt::CBIndex::c_29;
    constexpr uint32_t cb_done = tt::CBIndex::c_30;
    constexpr uint32_t cb_dq = tt::CBIndex::c_16;
    constexpr uint32_t cb_dk = tt::CBIndex::c_17;
    constexpr uint32_t cb_dv = tt::CBIndex::c_18;

    constexpr uint32_t chunks = Nt / Qt;
    constexpr uint32_t col_tiles = Nt * Dt;
    constexpr uint32_t qcol_tiles = Qt * Dt;
    constexpr uint32_t chunk_bias_tiles = Qt * Nt;

    generate_reduce_scaler(cb_scalar, identity_scalar_packed);
    generate_bcast_unary_scalar(cb_scale, scale_scalar_packed);
    for (uint32_t i = 0; i < Nt; ++i) {
        generate_bcast_col_scalar(cb_ones, identity_scalar_packed);
    }
    {
        cb_reserve_back(cb_zero, 1);
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_zero));
        for (uint32_t i = 0; i < (1024 * 2) / 4; ++i) {
            p[i] = 0;
        }
        cb_push_back(cb_zero, 1);
    }

    const auto dq_writer = TensorAccessor(dq_args, dq_addr, qkv_tile_bytes);
    const auto dk_writer = TensorAccessor(dk_args, dk_addr, kv_tile_bytes);
    const auto dv_writer = TensorAccessor(dv_args, dv_addr, kv_tile_bytes);
    const auto dbias_writer = TensorAccessor(dbias_args, dbias_addr, partial_tile_bytes);

    volatile tt_l1_ptr uint32_t* written =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    for (uint32_t c = 0; c < chunks; ++c) {
        for (uint32_t row = row_start; row < row_end; ++row) {
            const uint32_t base = (row * H + head) * col_tiles;

#ifdef PACKED_QKV
            // dq, dk and dv are one [B, 1, N, 3*H*d] buffer, the layout nlp_create_qkv_heads read
            // them out of: slot s, head h sits at channel tiles s*H*Dt + h*Dt. A head is whole
            // tiles, so only the page each tile goes to changes, and the head merges and the join
            // the tape would otherwise run after this kernel are not needed. Whole-query only:
            // a chunked dK/dV is a float32 running sum and has no place in a bf16 buffer.
            static_assert(Qt == Nt, "PACKED_QKV needs the whole-query kernel");
            (void)base;
            cb_wait_front(cb_dv, col_tiles);
            cb_wait_front(cb_dq, col_tiles);
            cb_wait_front(cb_dk, col_tiles);
            uint32_t qp = get_read_ptr(cb_dq);
            uint32_t kp = get_read_ptr(cb_dk);
            uint32_t vp = get_read_ptr(cb_dv);
            constexpr uint32_t W = 3 * H * Dt;
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                const uint32_t at = (row * Nt + nt) * W + head * Dt;
                for (uint32_t dt = 0; dt < Dt; ++dt) {
                    noc_async_write_tile(at + dt, dq_writer, qp);
                    noc_async_write_tile(at + H * Dt + dt, dk_writer, kp);
                    noc_async_write_tile(at + 2 * H * Dt + dt, dv_writer, vp);
                    qp += qkv_tile_bytes;
                    kp += qkv_tile_bytes;
                    vp += qkv_tile_bytes;
                }
            }
#else
            // The compute kernel pushes dV, then dQ, then dK.
            cb_wait_front(cb_dv, col_tiles);
            uint32_t vp = get_read_ptr(cb_dv);
            for (uint32_t i = 0; i < col_tiles; ++i) {
                noc_async_write_tile(base + i, dv_writer, vp);
                vp += kv_tile_bytes;
            }
            cb_wait_front(cb_dq, qcol_tiles);
            uint32_t qp = get_read_ptr(cb_dq);
            for (uint32_t i = 0; i < qcol_tiles; ++i) {
                noc_async_write_tile(base + c * qcol_tiles + i, dq_writer, qp);
                qp += qkv_tile_bytes;
            }
            cb_wait_front(cb_dk, col_tiles);
            uint32_t kp = get_read_ptr(cb_dk);
            for (uint32_t i = 0; i < col_tiles; ++i) {
                noc_async_write_tile(base + i, dk_writer, kp);
                kp += kv_tile_bytes;
            }
#endif
            noc_async_write_barrier();
            cb_pop_front(cb_dv, col_tiles);
            cb_pop_front(cb_dq, qcol_tiles);
            cb_pop_front(cb_dk, col_tiles);
            // Only this RISC-V writes it, so a plain store is the increment. It comes after the
            // barrier, which is the whole point: the reader reads this row back on the next chunk.
            *written = *written + 1;
        }

        cb_wait_front(cb_done, 1);
        {
            const uint32_t base = (group * H + head) * Nt * Nt + c * chunk_bias_tiles;
            uint32_t ptr = get_read_ptr(cb_dbias);
            for (uint32_t i = 0; i < chunk_bias_tiles; ++i) {
                noc_async_write_tile(base + i, dbias_writer, ptr);
                ptr += partial_tile_bytes;
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_done, 1);
    }
}
