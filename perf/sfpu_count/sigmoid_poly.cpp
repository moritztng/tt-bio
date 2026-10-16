// Entry points for perf/sfpu_count/count.py: each tile wrapper of sigmoid_poly.hpp, against stub LLK.
#include <cstdint>
extern volatile uint32_t __instrn_buffer[]; namespace ckernel { constexpr inline volatile uint32_t (__attribute__((rvtt_reg_ptr)) &instrn_buffer)[] = __instrn_buffer; }
#define TRISC_MATH 1
#define ALWI inline __attribute__((always_inline))
#define MATH(x) x
enum class VectorMode { RC = 4 };
template <bool A, typename F, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(F&& f, uint32_t, int, Args&&... a) {
    for (int face = 0; face < 4; face++) { f(static_cast<Args&&>(a)...); __asm__ volatile("" ::: "memory"); }
}
#include HDR
void use_sig(uint32_t i) { sigmoid_poly_tile<1>(i); }
void use_sig6(uint32_t i) { sigmoid_poly_tile<2>(i); }
void use_silu(uint32_t i) { silu_poly_tile(i); }
void use_silu6(uint32_t i) { silu_poly6_tile(i); }
#ifdef SIG_ROWS
void use_gate(uint32_t i) { gate_poly_tile<1, 1>(i); }
void use_gate6(uint32_t i) { gate_poly_tile<2, 1>(i); }
#else
void use_gate(uint32_t i) { gate_poly_tile<1>(i, i + 1); }
void use_gate6(uint32_t i) { gate_poly_tile<2>(i, i + 1); }
#endif
void use_rne(uint32_t i) { round_bf16_rne_tile(i); }
// Out of line, as the JIT's LTO emits them for a kernel with several call sites (the register-pressure worst case).
// count.py defines SIG_ROWS for the working header only (_sig_rows_ and the GAP-templated gate).
#ifdef SIG_ROWS
template void ckernel::sfpu::_sig_rows_<1, false, 0>();
template void ckernel::sfpu::_sig_rows_<1, true, 32>();
template void ckernel::sfpu::_sig_rows_<2, false, 0>();
template void ckernel::sfpu::_sig_rows_<2, true, 32>();
#endif
