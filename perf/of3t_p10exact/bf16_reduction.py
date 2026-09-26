"""Which half of a bf16 reduction destroys a weight gradient: the product, or the sum?

**THE APPLICATION TO `ttnn.sum` IS DEAD, killed on device 2026-09-26 by `of3t-p10exact` in a
6-second arm** (`perf/of3t_p10exact/sum_accumulator.py`, qb2 card 1 p300c). The CPU arithmetic
below is correct about a bf16 ACCUMULATOR in the abstract. `ttnn.sum` is not one:

    N        arithmetic                          rel L2      cos vs float64
    64       ttnn.sum, bf16 in, precise cfg      1.752e-03   0.9999986
    4096     ttnn.sum, bf16 in, precise cfg      1.741e-03   0.9999982
    147456   ttnn.sum, bf16 in, precise cfg      1.848e-03   0.9999980   <- FLAT across 2304x in N
    147456   ttnn.sum, fp32 in, precise cfg      7.565e-05   1.0000000
    147456   _sum_leading, fp32 in (add tree)    7.182e-08   1.0000000

**Flat in N means `ttnn.sum` already keeps an fp32 accumulator**; the residual 1.8e-03 is one bf16
rounding of the OUTPUT, not a sqrt(N) running-sum walk. `_sum_leading` on a bf16 input reads
bit-identically to plain `ttnn.sum`, confirming both which branch it takes and that the branch is
fine. **So the six token-axis sites (autograd.py 1283, 1341, 1343, 2709, 2769, 2771) are NOT the
defect they were claimed to be.**

WHERE THE CLAIM CAME FROM, because the mistake is instructive. `autograd.py:2698` says "a bf16
result means a bf16 running sum however precise the destination register is", measured as
"6.5e-02 relative L2 at K=4096 against 6.5e-03 at K=64, the sqrt(K) signature of a bf16
reduction". That comment is about a **MATMUL** (`_matmul` at 2705, fixed there with
`dtype=ttnn.float32`) and it is correct about the matmul. **Generalising it to `ttnn.sum` was mine
and it was wrong** -- and the sqrt(K) growth the comment reports is exactly what the arm above
shows `ttnn.sum` does NOT have.

WHAT SURVIVES, and it is worth a fraction of what was claimed. Passing fp32 INTO the reduction
still buys **24x** on the residual (7.565e-05 against 1.848e-03), and the fp32 add tree buys
another 1000x (7.182e-08). So `dtype=ttnn.float32` at those six sites is a real if modest
improvement, not a fix for anything catastrophic. Price it as such.

The CPU model below stands as a statement about bf16 accumulators generally. It was applied to the
wrong op.

--- original header follows ---

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
