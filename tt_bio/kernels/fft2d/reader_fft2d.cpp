// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// FFT reader. Streams the DFT matrix once per launch, then one image at a time.
//
// F is read once and never popped, so its DRAM cost amortises over NIMG images. That is the only
// reason this kernel processes a batch rather than a single image: at box 256 the matrix is 192
// tiles against an image's 128, so a one-image launch would spend more DRAM bandwidth on the
// constant than on the data.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_f = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t nf = get_compile_time_arg_val(3);      // 3 * NT * NT
    constexpr uint32_t ntile = get_compile_time_arg_val(4);   // 2 * NT * NT
    constexpr uint32_t nimg = get_compile_time_arg_val(5);
    constexpr auto f_args = TensorAccessorArgs<6>();
    constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();

    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(2);

    const auto fa = TensorAccessor(f_args, f_addr, tile_bytes);
    const auto xa = TensorAccessor(x_args, x_addr, tile_bytes);

    cb_reserve_back(cb_f, nf);
    uint32_t w = get_write_ptr(cb_f);
    for (uint32_t i = 0; i < nf; ++i) {
        noc_async_read_page(i, fa, w);
        w += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_f, nf);

    uint32_t page = page0;
    for (uint32_t img = 0; img < nimg; ++img) {
        cb_reserve_back(cb_x, ntile);
        w = get_write_ptr(cb_x);
        for (uint32_t i = 0; i < ntile; ++i) {
            noc_async_read_page(page + i, xa, w);
            w += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, ntile);
        page += ntile;
    }
}
