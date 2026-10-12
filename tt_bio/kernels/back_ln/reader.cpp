// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// back_ln reader: the trimul einsum's last channel chunk moved back to the last axis, and the
// channels before it, one output row block (fixed i, all Ct channel tiles) at a time.
// A unit is (it, jt, part): rows [part * R, part * R + R) of the 32 i rows of tile block (it, jt),
// R = 32 / P. Rows 0-15 of a tile are its first 1 KB (faces 0 and 1) and rows 16-31 the second, so
// with P = 2 each channel tile is one contiguous 1 KB read into scratch.
// Per i the reader pushes the Cp earlier channel tiles of row block (i, jt), already in the
// output layout in `prev` (rows j, cols c), to cb_prev, and the chunk's Clt gathered tiles
// M[c, j] = x_chunk[c, i, j] (rows c, cols j; the compute transposes them) to cb_in. The gather is
// reader_reblock_permute_back.cpp's, on half tiles.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t prev_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t beta_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t Ct = get_compile_time_arg_val(1);    // channel tiles of the output row
    constexpr uint32_t Cp = get_compile_time_arg_val(2);    // of them already in `prev`
    constexpr uint32_t P = get_compile_time_arg_val(3);     // row parts per tile block: 1, 2 or 4
    constexpr uint32_t SCALER = get_compile_time_arg_val(4);
    constexpr uint32_t EPS = get_compile_time_arg_val(5);
    constexpr uint32_t Clt = Ct - Cp;
    constexpr uint32_t R = 32 / P;
    constexpr uint32_t cb_in = 0, cb_prev = 1, cb_gamma = 2, cb_beta = 3, cb_scaler = 4, cb_eps = 5,
                       cb_stage = 24;
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto p_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, x_addr);
    const auto sp = TensorAccessor(p_args, prev_addr);

    constexpr uint32_t TILE_BYTES = 2048, FACE_BYTES = 512, FACE_ROW_BYTES = 32;
    // One part of a tile: R rows of face 0 then R rows of face 1 (P = 2: exactly the first or second
    // KB). For P = 4 a part is two 256 B pieces, both read.
    constexpr uint32_t PART_ROW_BYTES = R * FACE_ROW_BYTES;     // of one face, within one face row-half
    constexpr uint32_t STAGE_TILE = 2 * PART_ROW_BYTES;         // one channel's staged part

    generate_reduce_scaler(cb_scaler, SCALER);
    {
        cb_reserve_back(cb_eps, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(cb_eps, 1);
    }
    {
        const auto sg = TensorAccessor(g_args, gamma_addr);
        const auto sb = TensorAccessor(b_args, beta_addr);
        cb_reserve_back(cb_gamma, Ct);
        cb_reserve_back(cb_beta, Ct);
        uint32_t wg = get_write_ptr(cb_gamma), wb = get_write_ptr(cb_beta);
        for (uint32_t t = 0; t < Ct; ++t) {
            noc_async_read_page(t, sg, wg);
            noc_async_read_page(t, sb, wb);
            wg += TILE_BYTES;
            wb += TILE_BYTES;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Ct);
        cb_push_back(cb_beta, Ct);
    }

    cb_reserve_back(cb_stage, 1);
    const uint32_t stage = get_write_ptr(cb_stage);
    const uint64_t stage_noc = get_noc_addr(stage);
    constexpr uint32_t NtNt = Nt * Nt;
    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const uint32_t part = u % P;
        const uint32_t blk = u / P;
        const uint32_t it = blk / Nt, jt = blk - (blk / Nt) * Nt;
        // Stage the chunk's Clt * 32 channel tiles' part (row-half of face 0 and of face 1).
        {
            uint32_t page = it * Nt + jt, dst = stage;
            const uint32_t off = (part * R / 16) * 2 * FACE_BYTES + (part * R % 16) * FACE_ROW_BYTES;
            for (uint32_t c = 0; c < Clt * 32; ++c) {
                const uint64_t a = sx.get_noc_addr(page, off);
                noc_async_read(a, dst, PART_ROW_BYTES);
                noc_async_read(a + FACE_BYTES, dst + PART_ROW_BYTES, PART_ROW_BYTES);
                page += NtNt;
                dst += STAGE_TILE;
            }
            noc_async_read_barrier();
        }
        for (uint32_t il = 0; il < R; ++il) {
            const uint32_t i = it * 32 + part * R + il;
            if constexpr (Cp > 0) {
                cb_reserve_back(cb_prev, Cp);
                uint32_t w = get_write_ptr(cb_prev);
                const uint32_t page = (i * Nt + jt) * Ct;
                for (uint32_t t = 0; t < Cp; ++t) {
                    noc_async_read_page(page + t, sp, w);
                    w += TILE_BYTES;
                }
            }
            noc_async_read_one_packet_set_state(stage_noc, FACE_ROW_BYTES);
            for (uint32_t ct = 0; ct < Clt; ++ct) {
                cb_reserve_back(cb_in, 1);
                const uint32_t d = get_write_ptr(cb_in);
                uint32_t s0 = stage + ct * 32 * STAGE_TILE + il * FACE_ROW_BYTES;
                uint32_t s1 = s0 + PART_ROW_BYTES;
                // row c of M: face 0/1 for c < 16, face 2/3 for c >= 16
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
            if constexpr (Cp > 0) {
                noc_async_read_barrier();
                cb_push_back(cb_prev, Cp);
            }
        }
    }
}
