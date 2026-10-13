// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail T2CM reader: T2 (reader_ln.cpp) with x read from the einsum's channel-major output
// xcm [1, K, S, S] instead of the permuted [1, S, S, K], so `reblock_permute_back` never runs.
//
// The work unit is one cube (it, jt), St = S / 32: the 32 row tiles m = (it*32 + il)*St + jt,
// il in [0, 32), of the [S*S, K] matrix T2 consumes. Per cube:
//   1) the cube's K input tiles (channel c: page c*St*St + it*St + jt) as aligned 2 KB reads into the
//      scratch window (c_18), channel-ordered;
//   2) per block of Mb rows, per row il and channel tile ct, the gathered tile
//        T[cl, jl] = in_tile_{ct*32+cl}[il, jl] = x[it*32+il, jt*32+jl, ct*32+cl]
//      into c_6 (row-major over (mi, ct)), the gather of reader_reblock_permute_back.cpp verbatim: L1 -> L1
//      one-packet reads of one 16-element face row each, NOC state re-set before every block (the DRAM
//      reads below share the read command buffer and overwrite it). Compute transposes T into the real
//      x tile out[jl, cl], the tile the permute would have written;
//   3) the block's g_out rows (c_5) and z rows (c_7), page m*Nt + t, as T2 reads them.
// Once at entry, as T2: Wp (c_1), gamma / beta (c_8 / c_9), the 1/K scaler (c_10), the eps tile (c_11).
// z of a block is read before the writer writes it: the writer only gets that block after compute popped
// this read, and other blocks are other rows.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Mb = get_compile_time_arg_val(2);
    constexpr uint32_t SCALER = get_compile_time_arg_val(3);   // packed bf16 pair
    constexpr uint32_t EPS = get_compile_time_arg_val(4);      // fp32 bits
    constexpr uint32_t St = get_compile_time_arg_val(5);       // S / 32
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto bt_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto wp = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1));
    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2));
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(3));
    const auto gm = TensorAccessor(gm_args, get_common_arg_val<uint32_t>(4));
    const auto bt = TensorAccessor(bt_args, get_common_arg_val<uint32_t>(5));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    constexpr uint32_t gath_cb = tt::CBIndex::c_6;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t g_cb = tt::CBIndex::c_5;
    constexpr uint32_t z_cb = tt::CBIndex::c_7;
    constexpr uint32_t gamma_cb = tt::CBIndex::c_8;
    constexpr uint32_t beta_cb = tt::CBIndex::c_9;
    constexpr uint32_t scaler_cb = tt::CBIndex::c_10;
    constexpr uint32_t eps_cb = tt::CBIndex::c_11;
    constexpr uint32_t scratch_cb = tt::CBIndex::c_18;
    const uint32_t tb = get_tile_size(in1_cb);

    constexpr uint32_t TILE = 32, FACE = 16;
    constexpr uint32_t FACE_ROW_BYTES = FACE * 2;              // bf16
    constexpr uint32_t FACE_BYTES = FACE * FACE * 2;           // 512
    constexpr uint32_t K = Kt * TILE;
    constexpr uint32_t StSt = St * St;

    generate_reduce_scaler(scaler_cb, SCALER);
    {
        cb_reserve_back(eps_cb, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(eps_cb));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(eps_cb, 1);
    }
    cb_reserve_back(in1_cb, Kt * Nt);
    cb_reserve_back(gamma_cb, Kt);
    cb_reserve_back(beta_cb, Kt);
    uint32_t w = get_write_ptr(in1_cb);
    for (uint32_t t = 0; t < Kt * Nt; ++t, w += tb) noc_async_read_page(t, wp, w);
    w = get_write_ptr(gamma_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, gm, w);
    w = get_write_ptr(beta_cb);
    for (uint32_t t = 0; t < Kt; ++t, w += tb) noc_async_read_page(t, bt, w);
    noc_async_read_barrier();
    cb_push_back(in1_cb, Kt * Nt);
    cb_push_back(gamma_cb, Kt);
    cb_push_back(beta_cb, Kt);

    // The scratch window is this RISC's private staging area: reserved once, addressed directly.
    cb_reserve_back(scratch_cb, K);
    const uint32_t scratch = get_write_ptr(scratch_cb);
    // 64-bit: the high bits are this core's NOC x/y (see reader_reblock_permute_back.cpp).
    const uint64_t gather_state_addr = get_noc_addr(scratch);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t it = u / St;
        const uint32_t jt = u - it * St;

        // 1) the cube's K channel tiles
        {
            uint32_t page = it * St + jt;
            uint32_t dst = scratch;
            for (uint32_t c = 0; c < K; ++c) {
                noc_async_read_page(page, x, dst);
                page += StSt;
                dst += tb;
            }
        }
        noc_async_read_barrier();

        for (uint32_t il0 = 0; il0 < TILE; il0 += Mb) {
            // 2) gather Mb x Kt tiles, row il = il0 + mi, channel tile ct at slot mi*Kt + ct
            cb_reserve_back(gath_cb, Mb * Kt);
            const uint32_t gbase = get_write_ptr(gath_cb);
            noc_async_read_one_packet_set_state(gather_state_addr, FACE_ROW_BYTES);
            for (uint32_t mi = 0; mi < Mb; ++mi) {
                const uint32_t il = il0 + mi;
                const uint32_t src_il = scratch + (il / FACE) * 2 * FACE_BYTES + (il % FACE) * FACE_ROW_BYTES;
                for (uint32_t ct = 0; ct < Kt; ++ct) {
                    const uint32_t dst_tile = gbase + (mi * Kt + ct) * tb;
                    uint32_t s0 = src_il + ct * TILE * tb;   // face_w = 0 of channel ct*32
                    uint32_t s1 = s0 + FACE_BYTES;           // face_w = 1
                    uint32_t d0 = dst_tile;
                    uint32_t d1 = dst_tile + FACE_BYTES;
                    for (uint32_t cl = 0; cl < FACE; ++cl) {
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s1, d1);
                        s0 += tb;
                        s1 += tb;
                        d0 += FACE_ROW_BYTES;
                        d1 += FACE_ROW_BYTES;
                    }
                    d0 = dst_tile + 2 * FACE_BYTES;
                    d1 = dst_tile + 3 * FACE_BYTES;
                    for (uint32_t cl = FACE; cl < TILE; ++cl) {
                        noc_async_read_one_packet_with_state(s0, d0);
                        noc_async_read_one_packet_with_state(s1, d1);
                        s0 += tb;
                        s1 += tb;
                        d0 += FACE_ROW_BYTES;
                        d1 += FACE_ROW_BYTES;
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(gath_cb, Mb * Kt);

            // 3) g_out and z rows of the same Mb row tiles
            cb_reserve_back(g_cb, Mb * Nt);
            w = get_write_ptr(g_cb);
            for (uint32_t mi = 0; mi < Mb; ++mi) {
                const uint32_t m = (it * TILE + il0 + mi) * St + jt;
                for (uint32_t t = 0; t < Nt; ++t, w += tb) noc_async_read_page(m * Nt + t, g, w);
            }
            cb_reserve_back(z_cb, Mb * Nt);
            uint32_t wz = get_write_ptr(z_cb);
            for (uint32_t mi = 0; mi < Mb; ++mi) {
                const uint32_t m = (it * TILE + il0 + mi) * St + jt;
                for (uint32_t t = 0; t < Nt; ++t, wz += tb) noc_async_read_page(m * Nt + t, z, wz);
            }
            noc_async_read_barrier();
            cb_push_back(g_cb, Mb * Nt);
            cb_push_back(z_cb, Mb * Nt);
        }
    }
}
