#!/usr/bin/env python3
"""Two trunk arms against each other, in one metric, so a floor and a separation sit side by side.

    pairdiff.py <label> <a.pt> <b.pt> [--out report.json]

Per-parameter bit identity (as perf/of3t_verbinstall/armdiff.py), plus the concatenated
rel = ||a-b|| / ||b||, norm ratio and cos over the common tensors. On an A/A pair that rel IS the
floor; on the lever pair it is the separation, and the two are only comparable because they are
the same statistic over the same tensor set. Says nothing about accuracy. CPU only.
"""
from __future__ import annotations

import json
import math
import sys

import torch


def _grads(d):
    for k in ("grads", "gradients", "ours", "g"):
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]
    return d if isinstance(d, dict) else {}


def main() -> int:
    label, pa, pb = sys.argv[1:4]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else ""
    a = _grads(torch.load(pa, map_location="cpu"))
    b = _grads(torch.load(pb, map_location="cpu"))
    common = sorted(k for k in set(a) & set(b)
                    if torch.is_tensor(a[k]) and torch.is_tensor(b[k]) and a[k].shape == b[k].shape)
    bit, e2, aa, bb, ab = 0, 0.0, 0.0, 0.0, 0.0
    for k in common:
        x, y = a[k].double(), b[k].double()
        bit += int(torch.equal(x, y))
        e2 += float(((x - y) ** 2).sum()); aa += float((x * x).sum())
        bb += float((y * y).sum()); ab += float((x * y).sum())
    rel = math.sqrt(e2 / bb) if bb else None
    r = math.sqrt(aa / bb) if bb else None
    cos = ab / math.sqrt(aa * bb) if aa and bb else None
    rep = {"label": label, "a": pa, "b": pb,
           "compared": len(common), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
           "bit_identical": bit, "differing": len(common) - bit,
           "all_bit_identical": bit == len(common) and set(a) == set(b),
           "concatenated_rel_a_vs_b": rel, "norm_ratio_a_over_b": r, "cos": cos,
           "device_involved": False, "why_no_aiclk": "CPU only, compares two banked artifacts"}
    print("PAIRDIFF " + json.dumps(rep))
    if out:
        json.dump(rep, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
