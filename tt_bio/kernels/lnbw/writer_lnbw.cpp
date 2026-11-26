// Layer-norm backward writer: Wt tiles of dx per tile-row; with DO_W, then this core's Wt-tile partials of
// dgamma and dbeta (pages core * Wt + j of two [cores * 32, K] float32 tensors).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_row = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 16, cb_wo_gn = 20, cb_wo_b = 21;
    constexpr uint32_t DO_W = get_compile_time_arg_val(1);
    constexpr auto d_args = TensorAccessorArgs<2>();
    const auto sd = TensorAccessor(d_args, dst_addr);
    const uint32_t tb = get_tile_size(cb_out);
    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t page = (first_row + r) * Wt;
        cb_wait_front(cb_out, Wt);
        uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < Wt; ++j) { noc_async_write(rd, sd.get_noc_addr(page + j), tb); rd += tb; }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt);
    }
    if constexpr (DO_W) {
        constexpr auto p_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
        const uint32_t core = get_arg_val<uint32_t>(2);
        const auto sgn = TensorAccessor(p_args, get_common_arg_val<uint32_t>(1));
        const auto sb = TensorAccessor(p_args, get_common_arg_val<uint32_t>(2));
        const uint32_t ta = get_tile_size(cb_wo_gn);
        if (num_rows > 0) {
            cb_wait_front(cb_wo_gn, Wt);
            cb_wait_front(cb_wo_b, Wt);
            uint32_t ra = get_read_ptr(cb_wo_gn), rb = get_read_ptr(cb_wo_b);
            for (uint32_t j = 0; j < Wt; ++j) {
                noc_async_write(ra, sgn.get_noc_addr(core * Wt + j), ta);
                noc_async_write(rb, sb.get_noc_addr(core * Wt + j), ta);
                ra += ta; rb += ta;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_wo_gn, Wt);
            cb_pop_front(cb_wo_b, Wt);
        }
    }
}
