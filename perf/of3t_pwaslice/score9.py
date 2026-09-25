#!/usr/bin/env python3
"""of3t-pwaslice: per-tensor rel against float64 for the D262 weights, bf16 beside it.

    score9.py --f64 grads_f64.pt --bf16 grads_bf16.pt --bijection B.json --shapes S.json \
              --arm NAME=grad.pt ... --out F.json

Each arm is carried into upstream names through the bijection by of3t-fullstep64's own
`load_device`/`to_upstream`, so a figure here is the same figure `score.py` would compute for
that tensor. rel = ||g - g_64|| / ||g_64||, in float64; an arm that carries nothing reads 1.0.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_fullstep64"))
from score import load_device, to_upstream  # noqa: E402

D262 = re.compile(r"trunk\.msa_module\.blocks\.\d+\.pwa\.[mgo]_weight$")


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--f64", "--bf16", "--bijection", "--shapes", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--arm", action="append", default=[])
    a = ap.parse_args()
    bij = json.load(open(a.bijection))
    shapes = json.load(open(a.shapes))["shapes"]
    keys = sorted({k for k, pls in bij["placements"].items() for pl in pls
                   if D262.search(pl["device_path"])})
    ref = torch.load(a.f64, weights_only=False)
    bf = torch.load(a.bf16, weights_only=False)
    rel = lambda g, r: float((g.double() - r).norm() / r.norm())
    rows = {k: {"device": sorted(pl["device_path"] for pl in bij["placements"][k]),
                "f64_norm": float(ref[k].double().norm()),
                "bf16": rel(bf[k], ref[k].double())} for k in keys}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        up, _ = to_upstream(load_device(path, shapes), bij)
        for k in keys:
            rows[k][name] = rel(up[k], ref[k].double()) if k in up else None
    rec = {"f64": a.f64, "bf16": a.bf16, "n": len(keys), "tensors": rows}
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    for k, v in rows.items():
        print(k, {n: (round(x, 4) if isinstance(x, float) else x) for n, x in v.items()
                  if n != "device"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
