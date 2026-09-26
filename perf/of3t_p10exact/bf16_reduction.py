"""Which half of a bf16 reduction destroys a layer-norm weight gradient: the product, or the sum?

Pure numpy, no device, ~1 min on a CPU. Models `dgamma = sum_rows(g * norm)` exactly as
`tt_bio/autograd.py:1341` forms it, under three arithmetics, against a float64 reference.

MEASURED 2026-09-26 by `of3t-perf10` (N=147,456 is crop 384's 384x384 token axis, C=64):

    N        arithmetic                     rel L2      cos vs float64
    64       bf16 product, fp32 accumulator 2.674e-03   0.999996
    4096     bf16 product, fp32 accumulator 2.723e-03   0.999996
    147456   bf16 product, fp32 accumulator 2.859e-03   0.999996     <- FLAT in N. harmless.
    64       bf16 running sum               1.249e-02   0.999923
    4096     bf16 running sum               7.776e-02   0.997038
    147456   bf16 running sum               3.180e-01   0.954013     <- sqrt(N). this is the defect.
    147456   fp32 product, fp32 accumulator 6.179e-06   1.000000     <- the fix

CONCLUSION, and it corrects the first guess this row published: **rounding the PRODUCT to bf16 is
not the problem** -- that error is 2.7e-3 and does not grow with N, because fp32 destination
accumulation absorbs it. **The ACCUMULATOR is the problem**, and its error grows as sqrt(N).

That reproduces `autograd.py:2698`'s own banked measurement independently -- it recorded
"6.5e-02 relative L2 at K=4096 against 6.5e-03 at K=64, the sqrt(K) signature of a bf16
reduction", and the bf16-running-sum column here reads 7.8e-02 at 4096 and 1.2e-02 at 64. Two
routes, same mechanism, within a factor of two.

So the question for the device is narrow: does `ttnn.sum(bf16_in, compute_kernel_config=
precise_config())` keep an fp32 running sum, or does the bf16 OUTPUT dtype drive a bf16
accumulator? `autograd.py:2698` answers it in prose -- "a bf16 result means a bf16 running sum
however precise the destination register is" -- and fixes it with `dtype=ttnn.float32`, at ONE
site. Six token-axis sites do not have it: 1283, 1341, 1343, 2709, 2769, 2771.

NOT MODELLED, and it matters: the `cancel=True` rows below use a crude synthetic cancellation and
are unphysical (fp32 itself reads cos 0.35 there). Do not quote them. And cos 0.954 for one layer
does not by itself explain the trunk's measured cos 0.1796 -- 48 pairformer blocks compound
through the ACTIVATION gradient, which is the part this script does not model.
"""
import numpy as np

def bf16(x):
    """Round float32 to bfloat16, round-to-nearest-even."""
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) & 0xFFFF0000).view(np.float32)

def rel_l2(a, ref):
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))

def cos(a, ref):
    a, ref = a.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    return float(a @ ref / (np.linalg.norm(a) * np.linalg.norm(ref)))

def run(N, C, cancel, seed=0):
    rng = np.random.default_rng(seed)
    g    = rng.standard_normal((N, C)).astype(np.float32)
    norm = rng.standard_normal((N, C)).astype(np.float32)
    if cancel:                      # near-converged: per-column mean driven to ~0
        p = g * norm
        g = (g - p.mean(0) / np.where(norm.mean(0) == 0, 1, norm.mean(0))).astype(np.float32)
    ref = (g.astype(np.float64) * norm.astype(np.float64)).sum(0)

    # (a) current path: bf16 inputs -> bf16 product -> fp32 destination accumulation
    a = (bf16(bf16(g) * bf16(norm)).astype(np.float32)).sum(0, dtype=np.float32)
    # (b) a true bf16 running sum (what the file's 2698 comment describes)
    acc = np.zeros(C, np.float32); p = bf16(bf16(g) * bf16(norm))
    for i in range(N):
        acc = bf16(acc + p[i])
    # (c) proposed fix: fp32 product, fp32 accumulation
    c = (g.astype(np.float32) * norm.astype(np.float32)).sum(0, dtype=np.float32)
    return {"N": N, "cancel": cancel,
            "bf16prod_fp32acc": (rel_l2(a, ref), cos(a, ref)),
            "bf16prod_bf16acc": (rel_l2(acc, ref), cos(acc, ref)),
            "fp32prod_fp32acc": (rel_l2(c, ref), cos(c, ref))}

print(f"{'N':>8} {'cancel':>7} | {'bf16prod+fp32acc':>26} | {'bf16 running sum':>26} | {'fp32prod+fp32acc':>26}")
for N in (64, 4096, 147456):
    for cancel in (False, True):
        r = run(N, 64, cancel)
        f = lambda k: f"rel {r[k][0]:.3e} cos {r[k][1]:.6f}"
        print(f"{N:>8} {str(cancel):>7} | {f('bf16prod_fp32acc'):>26} | {f('bf16prod_bf16acc'):>26} | {f('fp32prod_fp32acc'):>26}")
