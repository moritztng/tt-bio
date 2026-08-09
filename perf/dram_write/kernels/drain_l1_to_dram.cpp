// W3 drain bench: pure L1 -> DRAM tile write, one data-movement kernel, nothing else in the way.
//
// The source is an L1 height-sharded tensor, so every read is core-local and the only NOC traffic
// this kernel makes is the DRAM write under study. The processor and the NOC are chosen host-side by
// the DataMovementConfigDescriptor, which is the point of the experiment: NOC_INDEX is a compile-time
// define and the firmware initialises per-RISC state for exactly that NOC, so switching NOCs from
// inside a kernel (as a source overlay would) hangs while switching it here does not.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_l1_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);   // this core's first global tile id
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);    // tiles this kernel issues
    const uint32_t src_off = get_arg_val<uint32_t>(4);      // first local tile index for this kernel

    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t stride = get_compile_time_arg_val(1);      // 1 = all tiles, 2 = every other
    constexpr uint32_t bar_every = get_compile_time_arg_val(2);   // barrier every N issues

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    uint32_t since_barrier = 0;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        const uint32_t j = src_off + i * stride;
        noc_async_write_tile(start_tile + j, s, src_l1_addr + j * tile_bytes);
        if (++since_barrier == bar_every) {
            noc_async_write_barrier();
            since_barrier = 0;
        }
    }
    noc_async_write_barrier();
}
