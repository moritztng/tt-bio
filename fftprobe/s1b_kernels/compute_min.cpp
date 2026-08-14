// Minimal bisect: no loop, no arithmetic. Copy in-tile 0 to the output CB. If this hangs, the
// hang is in the reader/writer/CB plumbing rather than in the timed loop.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mode = get_compile_time_arg_val(2);   // 0 = no loop, 1 = loop
    const uint32_t outer = get_arg_val<uint32_t>(0);

    binary_op_init_common(in_cb, in_cb, out_cb);
    cb_wait_front(in_cb, 2);

    if constexpr (mode == 1) {
        for (uint32_t i = 0; i < outer; ++i) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, 0, 0);
            copy_tile(in_cb, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
        }
    }

    cb_reserve_back(out_cb, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(in_cb);
    copy_tile(in_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);
    cb_pop_front(in_cb, 2);
}
