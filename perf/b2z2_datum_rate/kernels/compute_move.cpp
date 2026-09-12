// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Move-only compute kernel: the smallest thing that exercises the unpacker and the packer and
// nothing else. No CB traffic inside the timed loop, so the measured rate is the Tensix datum path,
// not a dataflow or dispatch artifact.
//
// MODE 0: copy_tile + pack_tile   -> the full L1 -> DST -> L1 round trip, GRAN tiles per acquire.
// MODE 1: copy_tile only          -> unpacker + FPU datacopy, packer idle. Output stays zero, which
//                                    is the negative control: a MODE 1 arm that produces data means
//                                    the define never reached the kernel.
// MODE 2: pack_tile only          -> one copy per group of GRAN packs, so the packer dominates by
//                                    GRAN-to-1 and the unpacker contributes 1/GRAN.
// MODE 3: neither                 -> the acquire/commit/wait/release barrier alone. This is the
//                                    instrument control: if MODE 3 does not come back near zero,
//                                    the compile-time arg never reached the kernel and every other
//                                    mode is reading the same compiled program.
// MODE 4: add_tiles + pack        -> the shape the trunk actually runs: TWO unpacks and one pack
//                                    per output tile, one into SrcA and one into SrcB.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t GRAN = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_SLOTS = get_compile_time_arg_val(4);
    constexpr uint32_t MODE = get_compile_time_arg_val(5);
    constexpr uint32_t in2_cb = get_compile_time_arg_val(6);

    const uint32_t reps = get_arg_val<uint32_t>(0);

    if constexpr (MODE == 4) {
        binary_op_init_common(in_cb, in2_cb, out_cb);
        add_tiles_init(in_cb, in2_cb);
    } else {
        unary_op_init_common(in_cb, out_cb);
        copy_tile_init(in_cb);
    }

    cb_wait_front(in_cb, NT);
    if constexpr (MODE == 4) {
        cb_wait_front(in2_cb, NT);
    }
    cb_reserve_back(out_cb, OUT_SLOTS);

    uint32_t o = 0;
    for (uint32_t r = 0; r < reps; ++r) {
        for (uint32_t i = 0; i < NT; i += GRAN) {
            tile_regs_acquire();
            if constexpr (MODE == 2) {
                copy_tile(in_cb, i, 0);
            } else if constexpr (MODE == 3) {
                // nothing: barrier only
            } else if constexpr (MODE == 4) {
                for (uint32_t j = 0; j < GRAN; ++j) {
                    add_tiles(in_cb, in2_cb, i + j, i + j, j);
                }
            } else {
                for (uint32_t j = 0; j < GRAN; ++j) {
                    copy_tile(in_cb, i + j, j);
                }
            }
            tile_regs_commit();

            tile_regs_wait();
            if constexpr (MODE != 1 && MODE != 3) {
                for (uint32_t j = 0; j < GRAN; ++j) {
                    pack_tile<true>(MODE == 2 ? 0 : j, out_cb, o);
                    o = (o + 1 == OUT_SLOTS) ? 0 : o + 1;
                }
            }
            tile_regs_release();
        }
    }

    cb_push_back(out_cb, OUT_SLOTS);
}
