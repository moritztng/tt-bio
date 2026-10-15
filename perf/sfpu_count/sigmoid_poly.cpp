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
void use_gate(uint32_t i) { gate_poly_tile<1>(i, i + 1); }
void use_gate6(uint32_t i) { gate_poly_tile<2>(i, i + 1); }
void use_rne(uint32_t i) { round_bf16_rne_tile(i); }
