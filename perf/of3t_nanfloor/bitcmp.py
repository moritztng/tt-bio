#!/usr/bin/env python3
"""Bit-identity between two gradient dumps, per tensor. A25's headline agreeing to 16 digits is
consistent with a tiny difference; this says whether there is one at all."""
import argparse
import json
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", nargs="+", required=True, help="label=a.pt:b.pt")
ap.add_argument("--out", default="")
a = ap.parse_args()

rep = {}
for spec in a.pairs:
    label, paths = spec.split("=", 1)
    pa, pb = paths.split(":")
    A = torch.load(pa, map_location="cpu")
    B = torch.load(pb, map_location="cpu")
    names = sorted(set(A) & set(B))
    same = 0
    worst = (0.0, None)
    for n in names:
        x, y = A[n].to(torch.float64), B[n].to(torch.float64)
        if x.shape != y.shape:
            worst = (float("inf"), n)
            continue
        d = float((x - y).abs().max())
        if d == 0.0:
            same += 1
        elif d > worst[0]:
            worst = (d, n)
    rep[label] = {"tensors": len(names), "only_in_a": sorted(set(A) - set(B))[:5],
                  "only_in_b": sorted(set(B) - set(A))[:5],
                  "bit_identical": same, "of": len(names),
                  "worst_abs_diff": worst[0], "worst_tensor": worst[1]}
    print(label, json.dumps(rep[label]), flush=True)
    del A, B
if a.out:
    json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
    print("wrote", a.out)
