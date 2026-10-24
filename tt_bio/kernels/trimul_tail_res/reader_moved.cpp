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
#include "../common/weight_mcast.hpp"
#ifndef T2MCAST
#define T2MCAST 0
#endif
// T2_DMABL (bench only, 0 in every model path): 1 skips the gather's L1 reads, 2 the stage's DRAM reads.
// T2_GROW (trimul_tail._t2_grow_fits): the gather fills a whole row's Clt tiles under one barrier, c_15 holds two.
#ifndef T2_DMABL
#define T2_DMABL 0
#endif
#ifndef T2_GROW
#define T2_GROW 0
#endif
// T2_S1K (trimul_tail.T2_S1K): at P = 2 the stage reads a part as one 1 KB read (its two faces sit side by side).
#ifndef T2_S1K
#define T2_S1K 0
#endif
// T2_SBIG (bench only, 0 in every model path): the stage reads its bytes as T2_SBIG-byte reads of contiguous DRAM,
// read k of unit u at tile page (u * reads + k) % 2048, so the reads rotate over the banks. The values are wrong; this
// prices a layout where a unit's stage sits in T2_SBIG-byte pages.
#ifndef T2_SBIG
#define T2_SBIG 0
#endif

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);     // output tiles per row (Wp columns)
    constexpr uint32_t St = get_compile_time_arg_val(2);     // N / 32, the pair's tile side
    constexpr uint32_t Cp = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);      // 1, 2 or 4
    constexpr uint32_t SCALER = get_compile_time_arg_val(5);
    constexpr uint32_t EPS = get_compile_time_arg_val(6);
    constexpr uint32_t GATE = get_compile_time_arg_val(7);
    constexpr uint32_t EXTRA = get_compile_time_arg_val(8);  // compute_ln.cpp's; bits 1 / 2 are read here
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto p_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto wg_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();
    constexpr auto bp_args = TensorAccessorArgs<bt_args.next_compile_time_args_offset()>();
    constexpr auto bg_args = TensorAccessorArgs<bp_args.next_compile_time_args_offset()>();
    constexpr uint32_t Clt = Kt - Cp;
    constexpr uint32_t R = 32 / P;

    const auto sx = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto sp = TensorAccessor(p_args, get_common_arg_val<uint32_t>(1));
    const auto wp = TensorAccessor(w_args, get_common_arg_val<uint32_t>(2));
    const auto wg = TensorAccessor(wg_args, get_common_arg_val<uint32_t>(3));
    const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(4));
    const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(5));
    const auto bp = TensorAccessor(bp_args, get_common_arg_val<uint32_t>(6));
    const auto bg = TensorAccessor(bg_args, get_common_arg_val<uint32_t>(7));
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_w = tt::CBIndex::c_1, cb_gamma = tt::CBIndex::c_8, cb_beta = tt::CBIndex::c_9,
                       cb_scaler = tt::CBIndex::c_10, cb_eps = tt::CBIndex::c_11, cb_in = tt::CBIndex::c_15,
                       cb_prev = tt::CBIndex::c_18, cb_stage = tt::CBIndex::c_24, cb_bias = tt::CBIndex::c_19;
    constexpr uint32_t NBIAS = (EXTRA & 1) + ((EXTRA >> 1) & 1);
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
    // T2MCAST (trimul_tail.T2_MCAST, not GATE / EXTRA): core 0 reads Wp, gamma, beta and multicasts them
    // (common/weight_mcast.hpp, common args 8..; per-core arg 2 marks the sender).
    static_assert(!T2MCAST || (!GATE && !EXTRA), "T2MCAST multicasts Wp alone");
    const bool sender = !T2MCAST || get_arg_val<uint32_t>(2);
    uint32_t w = get_write_ptr(cb_w);
    if (sender) {
        for (uint32_t t = 0; t < Kt * Nt; ++t, w += TILE_BYTES) noc_async_read_page(t, wp, w);
        if constexpr (GATE) {
            for (uint32_t t = 0; t < Kt * Nt; ++t, w += TILE_BYTES) noc_async_read_page(t, wg, w);
        }
        w = get_write_ptr(cb_gamma);
        for (uint32_t t = 0; t < Kt; ++t, w += TILE_BYTES) noc_async_read_page(t, gm, w);
        w = get_write_ptr(cb_beta);
        for (uint32_t t = 0; t < Kt; ++t, w += TILE_BYTES) noc_async_read_page(t, bt, w);
        if constexpr (NBIAS > 0) {
            // the [32, N] row-replicated biases, p's first
            cb_reserve_back(cb_bias, NBIAS * Nt);
            w = get_write_ptr(cb_bias);
            if constexpr (EXTRA & 1) {
                for (uint32_t t = 0; t < Nt; ++t, w += TILE_BYTES) noc_async_read_page(t, bp, w);
            }
            if constexpr (EXTRA & 2) {
                for (uint32_t t = 0; t < Nt; ++t, w += TILE_BYTES) noc_async_read_page(t, bg, w);
            }
        }
        noc_async_read_barrier();
    }
    if constexpr (T2MCAST) {
        if (sender) {
            const uint32_t addr[] = {get_write_ptr(cb_w), get_write_ptr(cb_gamma), get_write_ptr(cb_beta)};
            const uint32_t bytes[] = {Kt * Nt * TILE_BYTES, Kt * TILE_BYTES, Kt * TILE_BYTES};
            weight_mcast::send<8>(addr, bytes, 3);
        } else {
            weight_mcast::receive<8>();
        }
    }
    if constexpr (NBIAS > 0) cb_push_back(cb_bias, NBIAS * Nt);
    cb_push_back(cb_w, (1 + GATE) * Kt * Nt);
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
            if constexpr (T2_SBIG) {
                constexpr uint32_t NR = Clt * 32 * STAGE_TILE / (T2_SBIG ? T2_SBIG : 1);
                for (uint32_t k = 0; k < NR; ++k) {
                    noc_async_read(sx.get_noc_addr((u * NR + k) % 2048), stage + k * T2_SBIG, T2_SBIG);
                }
            }
            for (uint32_t c = 0; c < (T2_SBIG ? 0 : Clt * 32); ++c) {
                const uint64_t a = sx.get_noc_addr(page, off);
                if constexpr (T2_DMABL & 2) {
                    (void)a;
                } else if constexpr (P == 1) {
                    noc_async_read(a, dst, TILE_BYTES);         // the whole tile, faces in place
                } else if constexpr (P == 2 && T2_S1K) {
                    noc_async_read(a, dst, 2 * PART_ROW_BYTES);  // 16 rows are two whole faces, side by side
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
            if constexpr (T2_GROW) cb_reserve_back(cb_in, Clt);
            const uint32_t row0 = get_write_ptr(cb_in);
            for (uint32_t ct = 0; ct < Clt; ++ct) {
                if constexpr (!T2_GROW) cb_reserve_back(cb_in, 1);
                const uint32_t d = T2_GROW ? row0 + ct * TILE_BYTES : get_write_ptr(cb_in);
                // P = 1 stages whole tiles (row il: faces 2 * (il / 16) and + 1); P > 1 stages each
                // part's left face rows, then its right ones.
                uint32_t s0 = stage + ct * 32 * STAGE_TILE +
                              (P == 1 ? (il / 16) * 2 * FACE_BYTES + (il % 16) * FACE_ROW_BYTES : il * FACE_ROW_BYTES);
                uint32_t s1 = s0 + (P == 1 ? FACE_BYTES : PART_ROW_BYTES);
                for (uint32_t c = 0; c < ((T2_DMABL & 1) ? 0 : 32); ++c) {
                    const uint32_t row = (c / 16) * 2 * FACE_BYTES + (c % 16) * FACE_ROW_BYTES;
                    noc_async_read_one_packet_with_state(s0, d + row);
                    noc_async_read_one_packet_with_state(s1, d + row + FACE_BYTES);
                    s0 += STAGE_TILE;
                    s1 += STAGE_TILE;
                }
                if constexpr (!T2_GROW) {
                    noc_async_read_barrier();
                    cb_push_back(cb_in, 1);
                }
            }
            if constexpr (T2_GROW) {
                noc_async_read_barrier();
                cb_push_back(cb_in, Clt);
            }
        }
    }
}
