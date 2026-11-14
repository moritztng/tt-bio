// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gather_tiled writer: unit u's CT tiles are output pages u * CT .. u * CT + CT - 1 of the [.., R, W, C] TILE tensor.
// GT_ROWLIST: the units of a host list, as the reader walks them.
#include "api/dataflow/dataflow_api.h"
#ifdef GT_ROWLIST
#include "../common/row_list.hpp"
#endif

void kernel_main() {
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr uint32_t cb_out = 16;
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    const uint32_t ts = get_tile_size(cb_out);
#ifdef GT_ROWLIST
    const uint32_t me = get_arg_val<uint32_t>(2);
    const auto list = row_list_read<GT_ROWLIST, GT_LIST_BYTES>(me, get_common_arg_val<uint32_t>(1), get_write_ptr(6));
    const uint32_t nl = row_list_count(me, get_common_arg_val<uint32_t>(2), get_common_arg_val<uint32_t>(3));
    for (uint32_t i = 0; i < nl; ++i) {
        const uint32_t u = list[i];
#else
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
#endif
        cb_wait_front(cb_out, CT);
        uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < CT; ++t, rd += ts) {
            noc_async_write_page(u * CT + t, so, rd);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, CT);
    }
    noc_async_write_barrier();
}
