// Layer-norm backward reader: per tile-row, Wt tiles of x and of g; gamma once; the reduce
// scaler (1/K, exact: K is a power of two) and a full eps tile, generated here.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t first_row = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t DO_GAMMA = get_compile_time_arg_val(1);
    constexpr uint32_t SCALER = get_compile_time_arg_val(2);   // packed bf16 pair
    constexpr uint32_t EPS = get_compile_time_arg_val(3);      // fp32 bits
    constexpr uint32_t cb_x = 0, cb_g = 1, cb_gamma = 2, cb_scaler = 3, cb_eps = 4;

    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(x_args, x_addr);
    const auto sg = TensorAccessor(g_args, g_addr);

    generate_reduce_scaler(cb_scaler, SCALER);
    {
        cb_reserve_back(cb_eps, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
        for (uint32_t i = 0; i < 1024; ++i) p[i] = EPS;
        cb_push_back(cb_eps, 1);
    }
    if constexpr (DO_GAMMA) {
        const auto sgm = TensorAccessor(gm_args, gamma_addr);
        const uint32_t tb = get_tile_size(cb_gamma);
        cb_reserve_back(cb_gamma, Wt);
        uint32_t w = get_write_ptr(cb_gamma);
        for (uint32_t j = 0; j < Wt; ++j) { noc_async_read_page(j, sgm, w); w += tb; }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Wt);
    }
    const uint32_t tx = get_tile_size(cb_x), tg = get_tile_size(cb_g);
    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t page = (first_row + r) * Wt;
        cb_reserve_back(cb_x, Wt);
        cb_reserve_back(cb_g, Wt);
        uint32_t wx = get_write_ptr(cb_x), wg = get_write_ptr(cb_g);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_page(page + j, sx, wx);
            noc_async_read_page(page + j, sg, wg);
            wx += tx; wg += tg;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, Wt);
        cb_push_back(cb_g, Wt);
    }
}
