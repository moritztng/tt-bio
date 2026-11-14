#include "ckernel.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "ckernel_sfpu_silu.h"
using namespace ckernel::sfpu;
void silu_fp32() { calculate_silu<true, 8>(); }
void silu_bf16() { calculate_silu<false, 8>(); }
