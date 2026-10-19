#!/usr/bin/env python3
"""How much the card's matmul operand rounding costs the dense IPA layer, card-free.

A TT matmul reads its operands through the 19-bit source registers: an "fp32" matmul at HiFi4 sees a
10-bit mantissa (TF32), a bf16 one 7 bits; accumulation is fp32. This runs `ipa_fused.layer` in numpy
with every matmul operand rounded to m mantissa bits (round to nearest even) and the rest in float64,
and grades act / rotation / translation against the float64 golden. It answers whether the
point-folded head (key bias pw/2 |kp|^2 of a few hundred) survives the card's matmul precision.
Result, N 224, AF2 multimer weights, random act/z/frames (2026-10-19), rel L2 act / rotation / translation:
  tf32 every matmul       2.2e-3 / 3.1e-3 / 3.3e-4     bf16 every matmul       1.6e-2 / 1.7e-2 / 2.3e-3
  tf32 logits matmul only 1.2e-3 / 1.7e-3 / 1.8e-4     bf16 logits matmul only 9.1e-3 / 1.1e-2 / 1.4e-3
  floor: the EXACT layer with only its input act rounded to tf32 moves 9.8e-4 (bf16: 9.2e-3), and the
  golden explicit-distance form with act + weights rounded moves 1.5e-3 (bf16 1.0e-2). On these inputs
  the layer amplifies rounding ~4x by itself, and the point-folded chain costs ~1.5x the floor any tf32
  implementation pays. The expansion is not the precision risk; whether 2e-3 a layer is good enough is
  a structure-level question (Angstrom, real trunk inputs), not a logits one.
usage: ipa_fused_precision.py --params <npz> [--n 224]
"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipa_fused as F


def rnd(x, m):
    x = np.asarray(x, np.float64)
    e = np.floor(np.log2(np.maximum(np.abs(x), 1e-300)))
    q = 2.0 ** (e - m)
    return np.where(x == 0, 0.0, np.round(x / q) * q)


def backend(m, attn_only=False):
    class Q(F.NP):
        pass
    if m is None:
        return F.NP()
    r = lambda x: rnd(x, m)
    if not attn_only:
        Q.mm = staticmethod(lambda a, b: r(a) @ r(b))
        Q.linear = staticmethod(lambda x, w, b=None: r(x) @ r(w) + (0 if b is None else b))
    Q.bmm = staticmethod(lambda a, b: r(a) @ r(b))
    Q.bmm_t = staticmethod(lambda a, b: r(a) @ np.swapaxes(r(b), -1, -2))
    return Q()


def main(a):
    raw = np.load(a.params)
    p = {k: np.asarray(raw[k], np.float64) for k in raw.files if "/structure_module/" in k}
    k = F.pack(p)
    rng = np.random.default_rng(1)
    n, real = a.n, a.n - 4
    act = rng.standard_normal((n, F.CS)); z = rng.standard_normal((n, n, F.CZ))
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    quat = rng.standard_normal((n, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = F.G.quaternion_rotation(quat[:, 1] / quat[:, 0], quat[:, 2] / quat[:, 0], quat[:, 3] / quat[:, 0]).reshape(n, 9)
    t = rng.standard_normal((n, 3)) * 5.0
    o = F.NP()
    bias = F.pair_bias(o, k, z, mask)
    ref = F.layer(o, k, act, R, t, bias, z)
    for label, m, att in (("tf32 everywhere", 10, False), ("bf16 everywhere", 7, False),
                          ("tf32 attention bmm only", 10, True), ("bf16 attention bmm only", 7, True)):
        got = F.layer(backend(m, att), k, act, R, t, bias, z)
        rel = [float(np.linalg.norm(x - y) / np.linalg.norm(x)) for x, y in zip(ref, got)]
        print(f"{label:26s} act {rel[0]:.2e}  rotation {rel[1]:.2e}  translation {rel[2]:.2e}")
    # The golden's own explicit-distance form at the same rounding, for scale: the point term is not
    # expanded there, so this is the floor the expansion is compared with.
    print("max |kp|^2 per head (internal units):",
          np.round(max(float(np.max(np.sum(F.G.row_phase_in(act, R.reshape(n, 3, 3), t, p)['kp'] ** 2, -1))), 0), 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=224)
    main(ap.parse_args())
