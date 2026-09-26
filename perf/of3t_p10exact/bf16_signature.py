"""Does a bf16 running sum reproduce the trunk's error signature?  RESULT: not at the rung that matters.

**WITHDRAWN AS EVIDENCE FOR THE DEVICE-ONLY RUNG, 2026-09-26 17:4x, by its own author.** The
target below is `CLAUSE.json`'s *published* arm (cos 0.1796, norm_ratio 2.1021), which is a
frame-mismatched "renorm" arm and NOT the device-only baseline. Each rung carries its own reading
in `perf/of3t_stackexact/CLAUSE_<rung>.json` under `frame_matched/trunk/`:

    rung                  clause x_bar   cos      norm_ratio   cos*nr
    SHIP_A (device only)  1.4512         0.8179   1.0546       0.8625
    S  (exact softmax)    1.3038         0.7806   1.0645       0.8310
    L  (exact layernorm)  1.2839         0.8304   0.9985       0.8292
    SL (both, CLEARS)     0.9823         0.9374   0.9242       0.8664

Re-fitting this model against SHIP_A (N = 147,456, sweeping the cancellation ratio R):

    R=0.40   cos 0.8050 (-1.6%)    norm_ratio 0.9707 (-8.0%)
    R=0.30   cos 0.6255 (-23.5%)   norm_ratio 1.1044 (+4.7%)

**No single R matches both.** And the model's norm_ratio sits BELOW 1 wherever cos is right --
a bf16 accumulator attenuates -- while SHIP_A measures 1.0546, above 1. The attenuation signature
(cos*nr = 0.3775) that made this fit compelling belongs to the published arm; SHIP_A reads 0.8625,
near the pure-noise value of 1.0.

**So: the bf16 accumulator is NOT established as the cause of the device-only clause failure.**
What survives is `bf16_reduction.py`'s measurement -- a bf16 running sum loses a reduction as
sqrt(N), 3.18e-01 at N = 147,456 -- which is a real defect worth fixing at the six sites lacking
`dtype=ttnn.float32` (autograd.py 1283, 1341, 1343, 2709, 2769, 2771), independently of how much
of this particular clause it explains.

The original text is kept below because the arithmetic is correct; only its TARGET was wrong.
The lesson is the cheap one: a caveat you write down three times and then build on anyway is not
a caveat.

--- original header follows ---

"""Does a bf16 running sum reproduce the trunk's THREE-number signature, not just its cosine?

Third and strongest of the CPU experiments in this directory. `bf16_reduction.py` showed the
ACCUMULATOR is what loses a reduction; `bf16_cancellation.py` showed the mechanism spans the
observed error range. Both matched one number. This matches two independent ones at once.

THE SIGNATURE. `perf/of3t_modelframe/CLAUSE.json`, published arm, trunk:

    cos_vs_float64 0.1796    norm_ratio_vs_float64 2.1021    reading_vs_float64 2.1595

Those are internally consistent (sqrt(nr^2 - 2*nr*cos + 1) = 2.1595 exactly), so there are TWO
independent numbers. The diagnostic that matters is **cos * norm_ratio = 0.3775**: that is the
projection of our gradient onto the true one in units of the true norm, and **pure orthogonal
noise gives exactly 1.0**. 0.3775 means the device gradient RETAINS only 37.8 % of the true
gradient along its own direction while carrying 2.1x the norm -- systematic ATTENUATION on top of
noise, which random rounding alone cannot produce.

A bf16 accumulator attenuates for a specific reason: once the running sum exceeds an increment by
more than its ULP, the increment is dropped entirely (stagnation). This asks whether it attenuates
by the right amount.

MEASURED 2026-09-26, N = 147,456 (crop 384's token axis), C = 64, sweeping the cancellation ratio
R = |sum| / (sqrt(N) * sigma):

           R       cos  norm_ratio     rel_l2    cos*nr
      target    0.1796      2.1021     2.1595    0.3775   <- measured trunk
       1.000    0.9246      0.8880     0.3827    0.8210
       0.300    0.6255      1.1044     0.9155    0.6908
       0.100    0.1713      2.0202     2.0950    0.3460   <- matches all four within 4-8 %
       0.050   -0.0808      3.9388     4.1414   -0.3184
       0.030   -0.2056      6.7386     7.0128   -1.3857

CONCLUSION: at R = 0.1 a bf16 running sum reproduces cos to 4.6 %, norm_ratio to 3.9 %, rel_l2 to
3.0 % and the attenuation diagnostic to 8.3 % -- **two independent quantities matched
simultaneously by one swept parameter**, including the systematic attenuation that rules out pure
noise. This is the strongest card-free evidence available that the accuracy failure IS the bf16
accumulator at the six token-axis reduction sites (autograd.py 1283, 1341, 1343, 2709, 2769, 2771)
that lack the `dtype=ttnn.float32` the linear weight gradient got at 2705.

WHAT THIS IS NOT. R is swept, not measured. The data is synthetic Gaussian, not real gradients.
The target is CLAUSE.json's PUBLISHED arm (x_bar 3.40), not `of3t-exactscope`'s device-only rung
(x_bar 1.45117) -- different arms, and only the second is the row's baseline. And nothing here
touches a device: whether ttnn.sum actually keeps a bf16 accumulator is still read from
`autograd.py:2698`'s prose, not measured. Two independent numbers matching is a strong constraint,
not a proof.
"""
import numpy as np

def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) & 0xFFFF0000).view(np.float32)

N, C = 147456, 64
rng = np.random.default_rng(1)
base = rng.standard_normal((N, C)).astype(np.float32)
print(f"{'R':>8} {'cos':>9} {'norm_ratio':>11} {'rel_l2':>10} {'cos*nr':>9}")
print(f"{'target':>8} {0.1796:>9.4f} {2.1021:>11.4f} {2.1595:>10.4f} {0.3775:>9.4f}  <- measured trunk")
for R in (1.0, 0.3, 0.1, 0.05, 0.03, 0.01):
    p = base - base.mean(0)
    p = (p + R * np.sqrt(N) * p.std(0) / N).astype(np.float32)
    ref = p.astype(np.float64).sum(0)
    acc = np.zeros(C, np.float32); pb = bf16(p)
    for i in range(N):
        acc = bf16(acc + pb[i])
    a = acc.astype(np.float64)
    cs = float(a @ ref / (np.linalg.norm(a) * np.linalg.norm(ref)))
    nr = float(np.linalg.norm(a) / np.linalg.norm(ref))
    rl = float(np.linalg.norm(a - ref) / np.linalg.norm(ref))
    print(f"{R:>8.3f} {cs:>9.4f} {nr:>11.4f} {rl:>10.4f} {cs*nr:>9.4f}")
