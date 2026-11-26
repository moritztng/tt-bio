#!/usr/bin/env python3
"""Gradcheck for `ttnn.rms_norm` as RFD4's trunk issues it, through the tape's verb entry.

The forward under test is `tt_bio.rfd4.trunk._rms` itself, inside `taped_ttnn.tape()`: the
production call site, not a twin. Graded by `tt_bio.train.checks.gradcheck` (float64 reference
validated by central differences first). Arms: bf16 and fp32 production dtypes; `--break`
swaps the backward for the layer-norm one (centring left on), which is wrong for RMSNorm and
must FAIL; and an A/A control that the shipped verb and the taped forward agree bit for bit.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.train import checks as CK  # noqa: E402

EPS = 1e-5  # tt_bio/rfd4/trunk.py EPS
# (name, shape, affine): the three widths `_rms` sees in RFD4: token 768 (node norm, QK norm,
# recycle), cond 256 (AdaLN), pair 384 (pair-bias norm, no weight).
CASES = {"rms_token": ((2, 64, 768), True), "rms_cond": ((2, 64, 256), False),
         "rms_pair": ((1, 32, 32, 384), False)}


def ref_forward(t):
    y = t["x"] * torch.rsqrt((t["x"] * t["x"]).mean(-1, keepdim=True) + EPS)
    return y * t["w"] if "w" in t else y


def build(name, rng):
    shape, affine = CASES[name]
    raw = {"x": rng.standard_normal(shape) * 2.0 + 0.5}
    if affine:
        raw["w"] = rng.standard_normal((shape[-1],)) * 0.5 + 1.0
    return raw


def run(name, dt_name, seed, ttnn, ag, T, trunk, tt):
    dt = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[dt_name]
    tdt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dt_name]
    rng = CK.case_rng(seed, name)
    raw = build(name, rng)
    rounded = {k: torch.from_numpy(v).to(tdt).to(torch.float64) for k, v in raw.items()}
    ref = {k: v.clone().requires_grad_(True) for k, v in rounded.items()}
    wt = torch.from_numpy(rng.standard_normal(tuple(ref["x"].shape)))
    dev = tt.get_device()

    def up(v, shape=None):
        v = v.to(tdt) if shape is None else v.to(tdt).reshape(shape)
        return ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                           math_approx_mode=False, fp32_dest_acc_en=True,
                                           packer_l1_acc=True)
    wshape = (1, rounded["w"].shape[0]) if "w" in rounded else None
    xv = up(rounded["x"])
    wv = up(rounded["w"], wshape) if wshape else None
    # A/A: production verb, untaped, against the taped forward's value.
    shipped = ttnn.to_torch(trunk._rms(xv, wv, ckc)).to(torch.float64)
    leaves = {"x": ag.Tensor(xv, requires_grad=True)}
    if wv is not None:
        leaves["w"] = ag.Tensor(wv, requires_grad=True)
    with T.tape():
        out = trunk._rms(leaves["x"], leaves.get("w"), ckc)
    aa = bool(torch.equal(ttnn.to_torch(out.value).to(torch.float64), shipped))
    out.backward(seed=up(wt))
    got = {k: ttnn.to_torch(v.grad).to(torch.float64).numpy().reshape(rounded[k].shape)
           for k, v in leaves.items()}
    rep = CK.gradcheck(f"{name}/{dt_name}", ref_loss=lambda: (ref_forward(ref) * wt).sum(),
                       ref_params=list(ref.values()), tt_grads=got, op_class="reduction",
                       dtype=dt_name, n_probe=24, seed=seed)
    return {"case": name, "dtype": dt_name, "pass": rep.passed, "aa_bit_exact": aa,
            "fd_worst": rep.fd_worst, "fd_probed": rep.fd_probed,
            "grads": {k: {m: g[m] for m in ("rel_l2", "cos", "max_rel", "pass")}
                      for k, g in rep.grads.items()},
            "text": str(rep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="bfloat16,float32")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--break", dest="brk", action="store_true",
                    help="negative control: RMSNorm graded with the layer-norm backward")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import autograd as ag, taped_ttnn as T, tenstorrent as tt
    from tt_bio.rfd4 import trunk
    if a.brk:
        T._VERBS["rms_norm"] = lambda s, args, kw: ag._taped_layer_norm(s, args, kw, center=True)
    rows = [run(c, d, a.seed, ttnn, ag, T, trunk, tt) for d in a.dtypes.split(",") for c in CASES]
    for r in rows:
        print(r["text"], "| A/A bit-exact:", r["aa_bit_exact"], flush=True)
    rep = {"break": a.brk, "seed": a.seed, "rows": rows,
           "all_pass": all(r["pass"] and r["aa_bit_exact"] for r in rows)}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1))
    print("ALL_PASS" if rep["all_pass"] else "SOME_FAIL")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
