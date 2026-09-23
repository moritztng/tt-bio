#!/usr/bin/env python3
"""Two arm artifacts, per parameter: bit-identical, or the worst relative difference.

The frame scorer aggregates 2736 tensors into one mass-weighted number, and two arms can agree
there to seventeen digits while differing in tensors whose mass is too small to move it. For a
REPRODUCTION that aggregate is not enough, so this is the per-parameter claim with a worst case
behind it, which is what the campaign's gate asks of any "all parameters match".

    armdiff.py <mine.pt> <theirs.pt> [--out report.json]

Says nothing about accuracy. It says whether two arms computed the same gradient.
"""
from __future__ import annotations

import json
import sys

import torch


def _grads(d):
    for k in ("grads", "gradients", "ours", "g"):
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]
    return d if isinstance(d, dict) else {}


def main() -> int:
    mine, theirs = sys.argv[1], sys.argv[2]
    out = ""
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    a = _grads(torch.load(mine, map_location="cpu"))
    b = _grads(torch.load(theirs, map_location="cpu"))
    ka, kb = set(a), set(b)
    common = sorted(ka & kb)

    bit = 0
    diff = []
    for k in common:
        x, y = a[k], b[k]
        if not (torch.is_tensor(x) and torch.is_tensor(y)) or x.shape != y.shape:
            continue
        x, y = x.double(), y.double()
        if torch.equal(x, y):
            bit += 1
            continue
        d = (x - y).abs().max().item()
        ref = y.abs().max().item()
        diff.append({"tensor": k, "max_absdiff": d, "ref_max_abs": ref,
                     "rel": (d / ref) if ref > 0 else float("inf")})
    diff.sort(key=lambda r: -r["rel"])
    rep = {
        "what": "per-parameter comparison of two arm artifacts. `all_bit_identical` is the "
                "reproduction claim; the aggregate score cannot make it, because a tensor with "
                "too little mass to move the aggregate still differs.",
        "mine": mine, "theirs": theirs,
        "compared": len(common), "only_mine": sorted(ka - kb), "only_theirs": sorted(kb - ka),
        "bit_identical": bit, "differing": len(diff),
        "all_bit_identical": len(diff) == 0 and bit == len(common) and not (ka ^ kb),
        "worst": diff[:8],
    }
    print("ARMDIFF " + json.dumps({k: v for k, v in rep.items() if k != "worst"}))
    if out:
        with open(out, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
