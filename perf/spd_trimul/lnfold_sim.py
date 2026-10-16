#!/usr/bin/env python3
# Usage: python3 perf/spd_trimul/lnfold_sim.py  (CPU only, ~1 min). Result 10-12: the fold beats the incumbent at HiFi4 for
# any row mean; at HiFi3/HiFi2 its error grows with mean/std, so a fused LN-in needs a HiFi4 matmul.
# LN-in fold numerics: incumbent = bf16(LN(x)) @ bf16(W) with fp32 acc; fold = rstd*(x @ bf16(g*W) - mu*colsum) + beta@W.
# Operand mantissa truncation models matmul fidelity (bf16 keeps 7 explicit bits; HiFi2 ~ one operand to 5 bits, LoFi ~ 3).
import numpy as np, sys
rng = np.random.default_rng(0)
def bf16(a):
    a = np.asarray(a, np.float32).copy(); u = a.view(np.uint32)
    u += 0x7FFF + ((u >> 16) & 1); u &= 0xFFFF0000; return a
def trunc(a, bits):  # keep `bits` explicit mantissa bits (bf16 = 7)
    a = np.asarray(a, np.float32).copy(); u = a.view(np.uint32); u &= np.uint32(~((1 << (23 - bits)) - 1) & 0xFFFFFFFF); return a
R, C, H = 4096, 256, 1024
W = rng.standard_normal((C, H)) / np.sqrt(C); g = 1 + 0.2 * rng.standard_normal(C); b = 0.1 * rng.standard_normal(C)
print("mu/sigma  fid   rel_rms_incumbent  rel_rms_fold")
for ratio in tuple(map(float, sys.argv[1].split(","))) if len(sys.argv) > 1 else (0, 0.5, 2, 5, 20):
    x = bf16(ratio + rng.standard_normal((R, C)) * (1 + 0.3 * rng.standard_normal((R, 1)))).astype(np.float64)
    mu = x.mean(1, keepdims=True); rstd = 1 / np.sqrt(x.var(1, keepdims=True) + 1e-5)
    ref = (((x - mu) * rstd) * g + b) @ W
    for name, bits in (("hifi4", 7), ("hifi3", 6), ("hifi2", 5)):
        ln = bf16(((x - mu) * rstd) * g + b)
        inc = (trunc(ln, bits).astype(np.float32) @ bf16(W)).astype(np.float64)
        Wp = bf16(g[:, None] * W)
        xw = (trunc(x, bits).astype(np.float32) @ Wp).astype(np.float64)
        cs = Wp.astype(np.float64).sum(0)
        fold = rstd.astype(np.float32) * (xw - mu.astype(np.float32) * cs) + bf16(b @ W)
        e = lambda y: np.sqrt(((y - ref) ** 2).mean() / (ref ** 2).mean())
        print(f"{ratio:8} {name:6} {e(inc):.5f}            {e(fold):.5f}")
