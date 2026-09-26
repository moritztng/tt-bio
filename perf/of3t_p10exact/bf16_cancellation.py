"""Can a bf16 running sum alone explain the clause's cos 0.1796 and rel_l2 349, or is there a second defect?

Companion to `bf16_reduction.py`, which established that the ACCUMULATOR (not the bf16 product) is
what loses a reduction, at rel 3.18e-01 / cos 0.954 over N=147,456 rows. That is a real defect but
it does not by itself reach the trunk's measured `cos_vs_float64` 0.1796 or its worst tensor's
`rel_l2` 349.2 (`perf/of3t_modelframe/CLAUSE.json`). This asks what it takes.

The free parameter is CANCELLATION. `dgamma_j = sum_i g_ij * norm_ij` with g and norm roughly
independent gives zero-mean products, so |sum| ~ sqrt(N)*sigma. Call that R = 1. A near-converged
weight gradient cancels far harder than that. Sweeping R, N = 147,456, C = 64:

    R = |sum|/(sqrt(N)*sigma)   rel L2 (bf16 acc)        cos
                         1.000           3.827e-01     0.9246
                         0.300           9.155e-01     0.6255
                         0.100           2.095e+00     0.1713   <- the trunk measures cos 0.1796
                         0.030           7.013e+00    -0.2056
                         0.010           2.044e+01    -0.2685
                         0.003           6.735e+01    -0.2948
                                                                   rel_l2 349 needs R ~ 6e-4

CONCLUSION: **a bf16 running sum is sufficient on its own.** No second defect need be postulated.
At R = 0.1 it reproduces the trunk's composed cosine, and a single near-zero-gradient tensor at
R ~ 6e-4 reaches rel_l2 349. Both are ordinary for weight gradients in a model being trained.

**This is a consistency check, not a fit.** R was swept, not measured, and one parameter chosen to
land on one number proves little by itself. What it does establish is that the mechanism SPANS the
observed range -- which the earlier "cos 0.954 does not reach 0.1796" gap left open, and which a
second defect would have been needed to explain.

THE DISCRIMINATOR, for whoever holds the card: **measure R on the real tensors.** For each of the
clause's worst layer-norm weights, compute |sum| / (sqrt(N) * std(products)) from the float64
reference. R ~ 0.1 on the trunk and R ~ 1e-3 on the worst tensors confirms this whole chain. R ~ 1
kills it and says look elsewhere.
"""
import numpy as np

def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) & 0xFFFF0000).view(np.float32)

N, C = 147456, 64
rng = np.random.default_rng(1)
print(f"{'R = |sum|/(sqrt(N)*sigma)':>26} {'rel L2 (bf16 acc)':>19} {'cos':>10}")
base = rng.standard_normal((N, C)).astype(np.float32)
for R in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003):
    p = base - base.mean(0)                      # exact zero sum
    tgt = R * np.sqrt(N) * p.std(0)              # put back a controlled residual
    p = (p + tgt / N).astype(np.float32)
    ref = p.astype(np.float64).sum(0)
    acc = np.zeros(C, np.float32); pb = bf16(p)
    for i in range(N):
        acc = bf16(acc + pb[i])
    rel = float(np.linalg.norm(acc - ref) / np.linalg.norm(ref))
    a64, r64 = acc.astype(np.float64), ref
    cs = float(a64 @ r64 / (np.linalg.norm(a64) * np.linalg.norm(r64)))
    print(f"{R:>26.3f} {rel:>19.3e} {cs:>10.4f}")
