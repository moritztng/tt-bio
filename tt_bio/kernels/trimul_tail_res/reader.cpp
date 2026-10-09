// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// trimul tail, weights resident. Every core loads both [K, N] weights into c_1 once, then streams
// only its own rows of the activation(s) and, with RESID, of the residual z. No core forwards
// anything to another: the multicast chain of the 2D matmul it replaces is what the stage
// ablation found binding (~2.8 ms of a 3.4 ms pass at 736, every stage's work removed).
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

    const auto xa = TensorAccessor(xa_args, get_common_arg_val<uint32_t>(0));
    const auto wa = TensorAccessor(wa_args, get_common_arg_val<uint32_t>(1));
    const auto xb = TensorAccessor(xb_args, get_common_arg_val<uint32_t>(2));
    const auto wb = TensorAccessor(wb_args, get_common_arg_val<uint32_t>(3));
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(4));
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t z_cb = tt::CBIndex::c_7;
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
    }
}
