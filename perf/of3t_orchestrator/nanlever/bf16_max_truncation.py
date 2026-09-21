#!/usr/bin/env python3
"""Why the 5-op accurate softmax divides 0/0, and the exact condition under which it cannot.

`of3t-softgrad` measured on a p300c that `ttnn.max` comes back 1_755_648 ABOVE the true row
maximum on a row masked at -1e9, and that every exponent then underflows so the divide is 0/0
(perf/of3t_softgrad/FULLY_MASKED_ROW_OVERFLOW.json, 114_688 non-finite entries).

This script reproduces that number on a CPU with no device and no ttnn, from the hypothesis that
`ttnn.max` TRUNCATES its result to bf16 (round toward zero). If the hypothesis is right the
overshoot is not approximately 1.76e6, it is exactly 1_755_648, because -1e9 sits 1_755_648 above
the bf16 grid point below it. It is, which is what makes the mechanism a fact rather than a story.

The corollary is the part that decides the product question, and it is an invariant rather than an
observation: truncating a value that is ALREADY on the bf16 grid returns that value. So a row whose
elements all arrive as bf16 gets an EXACT maximum, d == 0 at the maximum element, exp == 1, and the
sum cannot be zero. The chain can only produce 0/0 on an fp32 input carrying more mantissa than
bf16 holds. That is the test a call site has to fail before it is exposed.
"""
import json
import math
import struct
import sys

F = lambda b: struct.unpack('<f', struct.pack('<I', b))[0]
I = lambda x: struct.unpack('<I', struct.pack('<f', float(x)))[0]


def bf16_trunc(x):
    """ttnn.max's observed behaviour: keep the top 7 fraction bits, discard the rest."""
    return F(I(x) & 0xFFFF0000)


def bf16_store(x):
    """What a bf16 TENSOR holds: round to nearest, ties to even."""
    b = I(x)
    lo, hi = b & 0xFFFF, b & 0xFFFF0000
    if lo > 0x8000 or (lo == 0x8000 and (hi >> 16) & 1):
        hi += 0x10000
    return F(hi)


F32_MIN_NORMAL = 1.1754943508222875e-38


def f32(x):
    """Round to fp32 and flush subnormals, which is what the SFPU does.

    Modelling this is not pedantry: it is the whole reason a floor at -88 does not work where one
    at -60 does. exp(-88) = 6.06e-39 is a fp32 SUBNORMAL, the device flushes it to zero, and the
    sum is zero again. In Python doubles it is an ordinary number and the -88 arm comes out finite,
    which disagrees with the device. The model has to carry the flush to reproduce both readings.
    """
    y = F(I(x))
    return 0.0 if 0.0 < abs(y) < F32_MIN_NORMAL else y


def chain(row, clamp=None):
    """max / subtract / exp / sum / divide, with `ttnn.max` truncating to bf16, all in fp32."""
    m = bf16_trunc(max(row))
    d = [f32(v - m) for v in row]
    if clamp is not None:
        d = [v if v > clamp else clamp for v in d]
    e = [f32(math.exp(v)) if v > -745.0 else 0.0 for v in d]
    s = 0.0
    for v in e:
        s = f32(s + v)
    if s == 0.0:
        return None, m, s          # 0/0
    return [f32(v / s) for v in e], m, s


out = {}

# 1. the measured overshoot, from first principles
true_max = -1e9
m = bf16_trunc(true_max)
out["truncation"] = {
    "fp32_mask_value": true_max,
    "bf16_truncated_max": m,
    "overshoot_above_true_max": m - true_max,
    "relative": (m - true_max) / abs(true_max),
    "softgrad_device_measurement": 1755648.0,
    "reproduces_exactly": (m - true_max) == 1755648.0,
}

# 2. a fully-masked row, fp32: the shipped chain, and the -60 floor
row_fp32 = [-1e9 + i * 1e-3 for i in range(128)]
p, m, s = chain(row_fp32)
out["fully_masked_fp32_shipped"] = {"finite": p is not None, "sum": s, "max_used": m}
p, m, s = chain(row_fp32, clamp=-60.0)
out["fully_masked_fp32_clamp60"] = {
    "finite": p is not None, "sum": s,
    "uniform": p is not None and max(abs(v - 1.0 / len(row_fp32)) for v in p) < 1e-6,
    "softgrad_device_measurement": "finite, 0 non-finite entries",
    "agrees_with_device": p is not None,
}
p, m, s = chain(row_fp32, clamp=-88.0)
out["fully_masked_fp32_clamp88"] = {
    "finite": p is not None,
    "softgrad_device_measurement": "not finite, 114688 non-finite entries",
    "agrees_with_device": p is None,
    "why": "exp(-88) = %.3e is below fp32's smallest normal 1.18e-38 and flushes to zero"
           % math.exp(-88),
}

# 3. the same row as a bf16 tensor holds it -- the shipped Pairformer sites
row_bf16 = [bf16_store(v) for v in row_fp32]
p, m, s = chain(row_bf16)
out["fully_masked_bf16_shipped"] = {
    "finite": p is not None, "sum": s,
    "max_is_exact": m == max(row_bf16),
    "uniform": p is not None and max(abs(v - 1.0 / len(row_bf16)) for v in p) < 1e-12,
    "note": "every element is already a bf16 grid point, so the truncation is the identity",
}

# 4. what the floor costs a live row: attention logits with a padded tail masked at -1e9
live = [bf16_store(v) for v in
        [0.3, -1.2, 2.7, -0.4, 1.1, -2.9, 0.0, 4.2] + [-1e9] * 120]
a, _, _ = chain(live)
b, _, _ = chain(live, clamp=-60.0)
worst = max(abs(x - y) for x, y in zip(a, b))
out["live_row_cost_of_the_floor"] = {
    "worst_absolute_change_in_a_weight": worst,
    "bound_exp_minus_60": math.exp(-60),
    "within_bound": worst <= math.exp(-60),
    "largest_weight": max(a),
    "relative_to_the_largest_weight": worst / max(a),
}

json.dump(out, sys.stdout, indent=1, sort_keys=True)
print()
