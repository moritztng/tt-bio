// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail T2, x read where the einsum left it (T2m): the last channel chunk channel-major,
// [1, Clt * 32, N, N], plus the Cp channel tiles earlier chunks already moved into `prev`
// ([1, N, N, Kt * 32], row tile m = i * Nt + jt). This replaces reblock_permute_back + T2's x read:
// the moved x is never written. Once: Wp (and Wg under GATE) into c_1, gamma / beta into c_8 / c_9,
// the 1/K scaler into c_10, an fp32 eps tile into c_11, as reader_ln.cpp. Then per unit (it, jt,
// part), back_ln/reader.cpp's stage and gather: R = 32 / P rows i of tile block (it, jt), per row
// the Cp prev tiles to c_18 and the Clt gathered tiles M[c, j] = x_chunk[c, i, j] to c_15 (the
// compute transposes them). g / xb and z rows are the writer's (writer_moved.cpp), so this RISC
// does nothing but the gather, which is the critical path.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "../common/ln_rows.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);     // output tiles per row (Wp columns)
    constexpr uint32_t St = get_compile_time_arg_val(2);     // N / 32, the pair's tile side
    constexpr uint32_t Cp = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);      // 1, 2 or 4
    constexpr uint32_t SCALER = get_compile_time_arg_val(5);
    constexpr uint32_t EPS = get_compile_time_arg_val(6);
    constexpr uint32_t GATE = get_compile_time_arg_val(7);
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto p_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto wg_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();
    constexpr uint32_t Clt = Kt - Cp;
    constexpr uint32_t R = 32 / P;

    const auto sx = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto sp = TensorAccessor(p_args, get_common_arg_val<uint32_t>(1));
    const auto wp = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2));
    const auto wg = TensorAccessor(wg_args, get_common_arg_val<uint32_t>(3));
    const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(4));
    const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(5));
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_w = tt::CBIndex::c_1, cb_gamma = tt::CBIndex::c_8, cb_beta = tt::CBIndex::c_9,
                       cb_scaler = tt::CBIndex::c_10, cb_eps = tt::CBIndex::c_11, cb_in = tt::CBIndex::c_15,
                       cb_prev = tt::CBIndex::c_18, cb_stage = tt::CBIndex::c_24;
    constexpr uint32_t TILE_BYTES = 2048, FACE_BYTES = 512, FACE_ROW_BYTES = 32;
    constexpr uint32_t PART_ROW_BYTES = R * FACE_ROW_BYTES;
    constexpr uint32_t STAGE_TILE = 2 * PART_ROW_BYTES;

    generate_reduce_scaler(cb_scaler, SCALER);
    {
        cb_reserve_back(cb_eps, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(cb_eps, 1);
    }
    cb_reserve_back(cb_w, (1 + GATE) * Kt * Nt);
    cb_reserve_back(cb_gamma, Kt);
    cb_reserve_back(cb_beta, Kt);
    uint32_t w = get_write_ptr(cb_w);
    for (uint32_t t = 0; t < Kt * Nt; ++t, w += TILE_BYTES) noc_async_read_page(t, wp, w);
    if constexpr (GATE) {
        for (uint32_t t = 0; t < Kt * Nt; ++t, w += TILE_BYTES) noc_async_read_page(t, wg, w);
    }
    w = get_write_ptr(cb_gamma);
    for (uint32_t t = 0; t < Kt; ++t, w += TILE_BYTES) noc_async_read_page(t, gm, w);
    w = get_write_ptr(cb_beta);
    for (uint32_t t = 0; t < Kt; ++t, w += TILE_BYTES) noc_async_read_page(t, bt, w);
    noc_async_read_barrier();
    cb_push_back(cb_w, (1 + GATE) * Kt * Nt);
    replicate_row0(get_write_ptr(cb_gamma), Kt, get_tile_size(cb_gamma));
    replicate_row0(get_write_ptr(cb_beta), Kt, get_tile_size(cb_beta));
    cb_push_back(cb_gamma, Kt);
    cb_push_back(cb_beta, Kt);

    cb_reserve_back(cb_stage, 1);
    const uint32_t stage = get_write_ptr(cb_stage);
    const uint64_t stage_noc = get_noc_addr(stage);
    constexpr uint32_t StSt = St * St;
    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const uint32_t part = u % P;
        const uint32_t blk = u / P;
        const uint32_t it = blk / St, jt = blk - (blk / St) * St;
        {
            uint32_t page = it * St + jt, dst = stage;
            const uint32_t off = (part * R / 16) * 2 * FACE_BYTES + (part * R % 16) * FACE_ROW_BYTES;
            for (uint32_t c = 0; c < Clt * 32; ++c) {
                const uint64_t a = sx.get_noc_addr(page, off);
                if constexpr (P == 1) {
                    noc_async_read(a, dst, TILE_BYTES);         // the whole tile, faces in place
                } else {
                    noc_async_read(a, dst, PART_ROW_BYTES);
                    noc_async_read(a + FACE_BYTES, dst + PART_ROW_BYTES, PART_ROW_BYTES);
                }
                page += StSt;
                dst += STAGE_TILE;
            }
            noc_async_read_barrier();
        }
        for (uint32_t il = 0; il < R; ++il) {
            const uint32_t i = it * 32 + part * R + il;
            if constexpr (Cp > 0) {
                cb_reserve_back(cb_prev, Cp);
                uint32_t d = get_write_ptr(cb_prev);
                const uint32_t page = (i * St + jt) * Kt;
                for (uint32_t t = 0; t < Cp; ++t, d += TILE_BYTES) noc_async_read_page(page + t, sp, d);
                noc_async_read_barrier();
                cb_push_back(cb_prev, Cp);
            }
            noc_async_read_one_packet_set_state(stage_noc, FACE_ROW_BYTES);
            for (uint32_t ct = 0; ct < Clt; ++ct) {
                cb_reserve_back(cb_in, 1);
                const uint32_t d = get_write_ptr(cb_in);
                // P = 1 stages whole tiles (row il: faces 2 * (il / 16) and + 1); P > 1 stages each
                // part's left face rows, then its right ones.
                uint32_t s0 = stage + ct * 32 * STAGE_TILE +
                              (P == 1 ? (il / 16) * 2 * FACE_BYTES + (il % 16) * FACE_ROW_BYTES : il * FACE_ROW_BYTES);
                uint32_t s1 = s0 + (P == 1 ? FACE_BYTES : PART_ROW_BYTES);
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t row = (c / 16) * 2 * FACE_BYTES + (c % 16) * FACE_ROW_BYTES;
                    noc_async_read_one_packet_with_state(s0, d + row);
                    noc_async_read_one_packet_with_state(s1, d + row + FACE_BYTES);
                    s0 += STAGE_TILE;
                    s1 += STAGE_TILE;
                }
                noc_async_read_barrier();
                cb_push_back(cb_in, 1);
            }
        }
    }
}
