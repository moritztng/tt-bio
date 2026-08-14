// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 1 emitting ROW-MAJOR strips, which is what makes stage 1 and stage 2 chainable on device.
//
// The z-collapse itself is unchanged from compute_zcollapse.cpp: W = sum_p mask_p * V_p, one
// mul_tiles per plane, exploiting the fact that mul_tiles ACCUMULATES into DST.
//
// What is new is the output layout. Stage 2's reader takes per-row contiguous windows out of an
// L1-resident plane at a per-row byte offset (reader_fslice.cpp), so it cannot be handed packed
// tiles: a TILE_LAYOUT source splits every logical row across two 16x16 faces. Stage 1 therefore
// has to untilize before it writes, and it has to do so in whole 32-row STRIPS rather than per
// tile -- a single 32x32 tile untilized on its own is 32 rows of 64 B landing at a 1024 B pitch in
// the destination plane, which fragments the write into 64 B pieces (S1c: 30 GB/s, does not
// pipeline). A strip of `strip_tiles` tiles untilized as one row-major block is 32 rows of
// `strip_tiles * 64` B, and at strip_tiles = 16 that is the full 512-wide padded plane row, so the
// whole strip is contiguous and the write goes out at the plane's row pitch.
//
// `pack_untilize_dest` is what does it, and its block_c_index parameter is the reason a strip does
// not need `strip_tiles` tiles live in DST at once: with full_ct_dim = strip_tiles, each call packs
// ONE tile into column band b of the same 32 x (32*strip_tiles) row-major output block.
//
// THE DST LAUNDERING IS NOT OPTIONAL. pack_untilize_dest cannot consume DST written by an FPU
// matmul/mul (projprobe/fslice_untilize.py: mode 10, matmul -> untilize, 1019 mismatches out of
// 1024; mode 8, copy_tile -> untilize, bit-exact). So the collapse packs each tile into a staging
// CB first, and the untilize pass brings it back with copy_tile. That is the same round trip
// compute_fslice.cpp's mode 12 uses, and it is why the strip is computed in two sweeps rather than
// one.
//
// `ncopy` is section 19's replication: the general shear needs the source held at all 8 sub-offsets
// because a per-row read offset is quantised to 8 bf16 elements from L1. COST PROBE -- the copies
// emitted here are unshifted, so the byte count is right and the contents of copies 1..7 are not.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_v = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(1);
    constexpr uint32_t cb_mid = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t nplane = get_compile_time_arg_val(4);
    constexpr uint32_t shift = get_compile_time_arg_val(5);
    constexpr uint32_t ncopy = get_compile_time_arg_val(6);
    constexpr uint32_t strip_tiles = get_compile_time_arg_val(7);
    // 0 = init/uninit the packer around every single untilize, which is the pattern mode 12 was
    // proven correct with. 1 = hoist it to once per strip. The difference is a named lever, not a
    // free choice: pack_untilize_dest_init is documented as an expensive MMIO write.
    constexpr uint32_t hoist_init = get_compile_time_arg_val(8);

    const uint32_t nstrip = get_arg_val<uint32_t>(0);

    binary_op_init_common(cb_v, cb_mask, cb_mid);
    cb_wait_front(cb_mask, nplane);

    for (uint32_t s = 0; s < nstrip; ++s) {
        cb_reserve_back(cb_mid, strip_tiles);
        for (uint32_t t = 0; t < strip_tiles; ++t) {
            cb_wait_front(cb_v, nplane);
            tile_regs_acquire();
            mul_tiles_init(cb_v, cb_mask);
            for (uint32_t p = 0; p < nplane; ++p) {
                mul_tiles(cb_v, cb_mask, p, p, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            // Explicit tile index: the whole strip is packed inside ONE reservation, so without it
            // every tile would land on slot 0 of the staging CB.
            pack_tile(0, cb_mid, t);
            tile_regs_release();
            cb_pop_front(cb_v, shift);
        }
        cb_push_back(cb_mid, strip_tiles);

        cb_wait_front(cb_mid, strip_tiles);
        if constexpr (hoist_init) {
            pack_untilize_dest_init<1, strip_tiles>(cb_out);
        }
        for (uint32_t q = 0; q < ncopy; ++q) {
            cb_reserve_back(cb_out, strip_tiles);
            for (uint32_t b = 0; b < strip_tiles; ++b) {
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_mid);
                copy_tile(cb_mid, b, 0);
                tile_regs_commit();
                tile_regs_wait();
                if constexpr (!hoist_init) {
                    pack_untilize_dest_init<1, strip_tiles>(cb_out);
                }
                pack_untilize_dest<1, strip_tiles>(cb_out, 1, b);
                if constexpr (!hoist_init) {
                    pack_untilize_uninit(cb_out);
                }
                tile_regs_release();
            }
            cb_push_back(cb_out, strip_tiles);
        }
        if constexpr (hoist_init) {
            pack_untilize_uninit(cb_out);
        }
        cb_pop_front(cb_mid, strip_tiles);
    }
}
