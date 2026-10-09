// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail, weights resident. Every core loads both [K, N] weights into c_1 once, then streams
// only its own rows of the activation(s) and, with RESID, of the residual z. No core forwards
// anything to another: the multicast chain of the 2D matmul it replaces is what the stage
// ablation found binding (~2.8 ms of a 3.4 ms pass at 736, every stage's work removed).
// With MASK (the gated in-projection) it also hands compute, per output row tile, the pair mask of
// those 32 rows as column 0 of a tile in c_6. Row r of the flattened [B * S * S] pair grid is
// m[b, x, y]; 32 consecutive rows share b and x (S % 32 == 0) and so are one row of one tile of the
// [B, S, S] mask, copied out of it here instead of materialising a column-shaped mask on the host.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Mb = get_compile_time_arg_val(2);
    constexpr uint32_t SHARED = get_compile_time_arg_val(3);
    constexpr uint32_t RESID = get_compile_time_arg_val(4);
    constexpr auto xa_args = TensorAccessorArgs<5>();
    constexpr auto wa_args = TensorAccessorArgs<xa_args.next_compile_time_args_offset()>();
    constexpr auto xb_args = TensorAccessorArgs<wa_args.next_compile_time_args_offset()>();
    constexpr auto wb_args = TensorAccessorArgs<xb_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<wb_args.next_compile_time_args_offset()>();
    constexpr uint32_t MASK = get_compile_time_arg_val(z_args.next_compile_time_args_offset());
    constexpr uint32_t St = get_compile_time_arg_val(z_args.next_compile_time_args_offset() + 1);
    constexpr auto m_args = TensorAccessorArgs<z_args.next_compile_time_args_offset() + 2>();

    const auto xa = TensorAccessor(xa_args, get_common_arg_val<uint32_t>(0));
    const auto wa = TensorAccessor(wa_args, get_common_arg_val<uint32_t>(1));
    const auto xb = TensorAccessor(xb_args, get_common_arg_val<uint32_t>(2));
    const auto wb = TensorAccessor(wb_args, get_common_arg_val<uint32_t>(3));
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(4));
    const auto mk = TensorAccessor(m_args, get_common_arg_val<uint32_t>(5));
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t z_cb = tt::CBIndex::c_7;
    constexpr uint32_t m_cb = tt::CBIndex::c_6;
    constexpr uint32_t mscratch_cb = tt::CBIndex::c_3;
    const uint32_t tb = get_tile_size(in0_cb);

    cb_reserve_back(in1_cb, 2 * Kt * Nt);
    uint32_t w = get_write_ptr(in1_cb);
    for (uint32_t t = 0; t < Kt * Nt; ++t, w += tb) noc_async_read_page(t, wa, w);
    for (uint32_t t = 0; t < Kt * Nt; ++t, w += tb) noc_async_read_page(t, wb, w);
    noc_async_read_barrier();
    cb_push_back(in1_cb, 2 * Kt * Nt);

    for (uint32_t b = 0; b < nblocks; ++b) {
        const uint32_t m = m0 + b * Mb;
        cb_reserve_back(in0_cb, Mb * Kt);
        w = get_write_ptr(in0_cb);
        for (uint32_t t = 0; t < Mb * Kt; ++t, w += tb) noc_async_read_page(m * Kt + t, xa, w);
        noc_async_read_barrier();
        cb_push_back(in0_cb, Mb * Kt);
        if constexpr (!SHARED) {
            cb_reserve_back(in0_cb, Mb * Kt);
            w = get_write_ptr(in0_cb);
            for (uint32_t t = 0; t < Mb * Kt; ++t, w += tb) noc_async_read_page(m * Kt + t, xb, w);
            noc_async_read_barrier();
            cb_push_back(in0_cb, Mb * Kt);
        }
        if constexpr (RESID) {
            // z rows of this block, read before the writer overwrites them: the writer only
            // writes block b after compute consumed this read.
            cb_reserve_back(z_cb, Mb * Nt);
            w = get_write_ptr(z_cb);
            for (uint32_t t = 0; t < Mb * Nt; ++t, w += tb) noc_async_read_page(m * Nt + t, z, w);
            noc_async_read_barrier();
            cb_push_back(z_cb, Mb * Nt);
        }
        if constexpr (MASK) {
            // Row tile R covers rows 32R..32R+31 = (b, x, y0..y0+31): row x % 32 of mask tile
            // (b, x / 32, y0 / 32). bf16 tile = four 16x16 faces, row-major inside each face.
            const uint32_t scratch = get_write_ptr(mscratch_cb);
            cb_reserve_back(m_cb, Mb);
            w = get_write_ptr(m_cb);
            for (uint32_t i = 0; i < Mb; ++i, w += tb) {
                const uint32_t R = m + i;
                const uint32_t b_ = R / (St * St * 32);
                const uint32_t rem = R % (St * St * 32);
                const uint32_t x = rem / St;
                const uint32_t yt = rem % St;
                noc_async_read_page(b_ * St * St + (x / 32) * St + yt, mk, scratch);
                noc_async_read_barrier();
                const uint32_t r = x % 32;
                volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
                volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w);
                for (uint32_t c = 0; c < 32; ++c) {
                    dst[((c >> 4) * 2) * 256 + (c & 15) * 16] =
                        src[((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)];
                }
            }
            cb_push_back(m_cb, Mb);
        }
    }
}
