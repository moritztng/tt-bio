// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 1 writer for the chained pipeline: one row-major strip out, at the destination plane's row
// pitch.
//
// The compute kernel hands over a 32 x (32*strip_tiles) row-major block. The destination is the
// replicated W plane stage 2 reads, laid out one padded plane row per page with the 8 sub-offset
// copies interleaved PER ROW (section 19 measured that at 1.89x over blocking them), so plane row
// R of copy q is page R*ncopy + q. A strip therefore goes out as `nrows` writes of one full row
// each, which at box 256 is 1024 B a piece -- coarse enough to stay in the bulk regime and the
// largest a per-row-interleaved replication permits.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t nrows = get_compile_time_arg_val(2);
    constexpr uint32_t ncopy = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(4);
    constexpr uint32_t nstrip_wrap [[maybe_unused]] = get_compile_time_arg_val(5);
    constexpr auto dst_args = TensorAccessorArgs<6>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t nstrip = get_arg_val<uint32_t>(1);
    const uint32_t strip0 = get_arg_val<uint32_t>(2);

    const auto d = TensorAccessor(dst_args, dst_addr, row_bytes);

    // The W buffer holds exactly one strip per core and each core owns its own, so a batch longer
    // than one strip per core rewrites it in place -- which is what a real pipeline does anyway,
    // since each block of directions reuses the same L1 W buffer. Stepping the strip index with the
    // batch instead was a bug: it made core c write strips c..c+nstrip-1, so cores overlapped on the
    // same strip and the last of them wrote past the end of the buffer. Same bytes and same
    // transaction count either way, but the contents were then a race.
    const uint32_t base = strip0 * nrows * ncopy;
    for (uint32_t s = 0; s < nstrip; ++s) {
        for (uint32_t q = 0; q < ncopy; ++q) {
            cb_wait_front(cb_out, tiles_per_block);
            uint32_t rp = get_read_ptr(cb_out);
            uint32_t pg = base + q;
            for (uint32_t r = 0; r < nrows; ++r) {
                noc_async_write_page(pg, d, rp);
                rp += row_bytes;
                pg += ncopy;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, tiles_per_block);
        }
    }
}
