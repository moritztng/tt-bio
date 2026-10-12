"""Fit and check sigmoid_poly.hpp's degree-6 silu: h(a) = sum c_k a^k (k = 1..6), a = min(|x|, X), fitted to
sigmoid(a) - 0.5 by iteratively reweighted least squares toward minimax, each coefficient rounded to fp16 in turn
(so SFPI loads it in one SFPLOADI) with the rest refitted. Then the kernel's float32 Horner order is emulated over
x in [-60, 60] and compared with float64. CPU only."""
import numpy as np

X, DEG = 7.0, 6
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def fit(fixed):
    a = np.linspace(0, X, 20001)
    A = np.stack([a ** k for k in range(1, DEG + 1)], 1)
    n = len(fixed)
    y = sig(a) - 0.5 - (A[:, :n] @ np.array(fixed) if n else 0)
    w = np.ones_like(a)
    for _ in range(80):
        c, *_ = np.linalg.lstsq(A[:, n:] * w[:, None], y * w, rcond=None)
        e = np.abs(A[:, n:] @ c - y)
        w = w * (1 + e / e.max()) ** 2
        w /= w.mean()
    return list(fixed) + list(c)


coef = []
for k in range(DEG):
    coef.append(float(np.float16(fit(coef)[k])))
print("c1..c6 =", [repr(v) for v in coef])

x = np.linspace(-60, 60, 2000001).astype(np.float32)
a = np.minimum(np.abs(x), np.float32(X))
c = [np.float32(v) for v in coef]
h = c[-1]
for k in range(DEG - 2, -1, -1):
    h = (h * a + c[k]).astype(np.float32)
h = (h * a).astype(np.float32)
s = np.copysign(h, x) + np.float32(0.5)
ref = sig(x.astype(np.float64))
print("sigmoid max abs err %.2e" % np.abs(s - ref).max())
pos = x > 1
print("silu max rel err, x > 1: %.2e" % (np.abs(s - ref)[pos] / ref[pos]).max())
print("silu max abs err, x < 0: %.2e" % np.abs(x * (s - ref))[x < 0].max())
