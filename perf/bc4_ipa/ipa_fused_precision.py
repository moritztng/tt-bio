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
Real inputs (`dump_sm_inputs.py`, BindCraft 2 PD-L1 draw, N 192, 175 real, both recycles), the whole
8-layer stack from identity frames, CA deviation against float64 (position_scale 20 to A):
  tf32 every matmul  CA RMSD 0.003 / 0.004 A, max 0.010 A, act 3.3e-3
  bf16 every matmul  CA RMSD 0.048 / 0.024 A, max 0.148 A, act 2.0e-2 / 2.6e-2
  (exact stack, single + pair rounded to tf32 once: 0.0002 / 0.021 A)
Against the 0.60 A bar the fp32 build has ~150x headroom on the forward and bf16 ~12x on the worst residue.
usage: ipa_fused_precision.py --params <npz> [--n 224]
       ipa_fused_precision.py --params <npz> --inputs <dump_sm_inputs.py npz>   (the 8-layer stack on real inputs)
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


def stack(o, k, p, single, pair, mask, layers=8):
    """generate_monomer_rigids without sidechains: LN + initial projection, identity frames, 8 layers."""
    act = F.G.layer_norm(single, p, "alphafold/alphafold_iteration/structure_module/single_layer_norm")
    act = F.G.linear(act, p, "alphafold/alphafold_iteration/structure_module/initial_projection")
    z = F.G.layer_norm(pair, p, "alphafold/alphafold_iteration/structure_module/pair_layer_norm")
    n = act.shape[0]
    R, t = np.tile(np.eye(3).reshape(1, 9), (n, 1)), np.zeros((n, 3))
    bias = F.pair_bias(F.NP(), k, z, mask)
    for _ in range(layers):
        act, R, t = o_layer(o, k, act, R, t, bias, z)
    return act, R, t


def o_layer(o, k, act, R, t, bias, z):
    return F.layer(o, k, act, R, t, bias, z)


def on_real_inputs(a, p, k):
    d = np.load(a.inputs)
    for i in range(len([f for f in d.files if f.startswith("single_")])):
        single, pair, mask = (np.asarray(d[f"{f}_{i}"], np.float64) for f in ("single", "pair", "seq_mask"))
        real_ = mask > 0
        ref = stack(F.NP(), k, p, single, pair, mask)
        for label, m in (("tf32 every matmul", 10), ("bf16 every matmul", 7), ("tf32 input only", None)):
            if m is None:     # the floor: exact stack, single and pair rounded to tf32 once
                got = stack(F.NP(), k, p, rnd(single, 10), rnd(pair, 10), mask)
            else:
                got = stack(backend(m), k, p, single, pair, mask)
            dca = np.linalg.norm((got[2] - ref[2])[real_], axis=-1) * 20.0      # position_scale 20: A
            rel = float(np.linalg.norm(got[0] - ref[0]) / np.linalg.norm(ref[0]))
            print(f"call {i} {label:18s} CA RMSD {np.sqrt((dca ** 2).mean()):.4f} A  max {dca.max():.4f} A  act rel L2 {rel:.2e}",
                  flush=True)


def main(a):
    raw = np.load(a.params)
    p = {k: np.asarray(raw[k], np.float64) for k in raw.files if "/structure_module/" in k}
    k = F.pack(p)
    if a.inputs:
        return on_real_inputs(a, p, k)
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
    ap.add_argument("--inputs")
    main(ap.parse_args())
