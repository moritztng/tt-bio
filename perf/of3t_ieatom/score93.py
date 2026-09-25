#!/usr/bin/env python3
"""of3t-ieatom: the 93 input-embedder atom-encoder tensors against float64, bf16 beside.

    score93.py --f64 grads_f64.pt --bf16 grads_bf16.pt --bijection B.json --shapes S.json \
               --arm NAME=grad.pt ... --out F.json

The 93 are BIJECTION_DN.json's `upstream_unplaced` under `input_embedder.`. Each arm goes into
upstream names through of3t-fullstep64's own `load_device`/`to_upstream`, so a figure here is the
figure `score.py` would give that tensor. rel = ||g - g_64|| / ||g_64|| in float64, per tensor
and over the 93 concatenated; an arm that carries nothing reads 1.0.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_fullstep64"))
from score import load_device, to_upstream  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("--f64", "--bf16", "--bijection", "--shapes", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--arm", action="append", default=[])
    a = ap.parse_args()
    dn = json.loads((HERE.parent / "of3t_denoise/BIJECTION_DN.json").read_text())
    keys = sorted(k for k in dn["upstream_unplaced"] if k.startswith("input_embedder."))
    bij = json.load(open(a.bijection))
    shapes = json.load(open(a.shapes))["shapes"]
    ref = {k: torch.load(a.f64, weights_only=False)[k].double() for k in keys}
    bf = torch.load(a.bf16, weights_only=False)
    arms = {"bf16": {k: bf[k].double() for k in keys}}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        up, _ = to_upstream(load_device(path, shapes), bij)
        arms[name] = {k: up[k] for k in keys if k in up}
    mass = sum(float(ref[k].pow(2).sum()) for k in keys)
    rec = {"f64": a.f64, "bf16": a.bf16, "bijection": a.bijection, "n": len(keys),
           "f64_mass": mass, "arms": {}, "tensors": {}}
    for name, g in arms.items():
        dd = sum(float((g.get(k, torch.zeros_like(ref[k])) - ref[k]).pow(2).sum()) for k in keys)
        rels = {k: float((g[k] - ref[k]).norm() / ref[k].norm()) if k in g else 1.0 for k in keys}
        rec["arms"][name] = {"placed": sum(k in g for k in keys),
                             "nonzero": sum(k in g and bool(g[k].abs().sum() > 0) for k in keys),
                             "rel_concat": (dd / mass) ** 0.5,
                             "rel_median": statistics.median(rels.values())}
        for k in keys:
            rec["tensors"].setdefault(k, {"f64_mass_fraction": float(ref[k].pow(2).sum()) / mass})
            rec["tensors"][k][name] = rels[k]
    for name in arms:
        if name != "bf16":
            rec["arms"][name]["at_or_better_than_bf16"] = sum(
                v[name] <= v["bf16"] for v in rec["tensors"].values())
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["arms"]), flush=True)
    top = sorted(rec["tensors"].items(), key=lambda kv: -kv[1]["f64_mass_fraction"])[:5]
    for k, v in top:
        print(k, {n: round(x, 4) for n, x in v.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
