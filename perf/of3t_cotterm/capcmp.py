#!/usr/bin/env python3
"""of3t-cotterm: tensor-by-tensor identity between two files, with the host in the artifact.

An identical SCORE is not an identity: `of3t-trunkopclass` found a lever that scored the same
and was bit-identical, and only a direct comparison separates those. This is the A/A
determinism floor and the capture-perturbation control, both.

`host` is emitted by the WRITER, not left to a brief line. `of3t-apbleaf` concluded with five
artifacts that carried no host field and the D155 guard stopped the whole composition for every
row in the campaign; they are now on a shrink-only ratchet (D235). The same applies to the board
class and, where the artifact is a timing, the AICLK.

Handles both shapes this row produces:
  gradient files  {"grads": {param: tensor}} or a bare {param: tensor}
  capture files   {"sites": {block: {field: tensor}}} or {"sites": [ {field: tensor} ]}
"""
from __future__ import annotations

import argparse
import json
import os

import torch

PRE = "pairformer_stack.blocks."


def flatten(path):
    """Every tensor in the file, keyed by a path a reader can locate."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    if isinstance(d, dict) and "sites" in d:
        sites = d["sites"]
        it = (sites.items() if isinstance(sites, dict)
              else enumerate(sites))
        for i, e in it:
            key = (e.get("gamma_path") or str(i)) if isinstance(e, dict) else str(i)
            for f, v in (e.items() if isinstance(e, dict) else []):
                if torch.is_tensor(v):
                    out[f"{key}|{f}"] = v
        return out
    if isinstance(d, dict) and "grads" in d and isinstance(d["grads"], dict):
        d = d["grads"]
    for k, v in d.items():
        if torch.is_tensor(v):
            kk = k if k.startswith(PRE) else (PRE + k if k.split(".")[0].isdigit() else k)
            out[kk] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--board", default="p150a Blackhole")
    ap.add_argument("--card", default="")
    ap.add_argument("--aiclk", default="", help="the AICLK summary line sampled DURING the arm")
    ap.add_argument("--out", required=True)
    z = ap.parse_args()

    A, B = flatten(z.a), flatten(z.b)
    keys = sorted(set(A) & set(B))
    moved, worst, worst_k = 0, 0.0, None
    for k in keys:
        x = A[k].to(torch.float64).reshape(-1)
        y = B[k].to(torch.float64).reshape(-1)
        if x.shape != y.shape or not torch.equal(x, y):
            moved += 1
            r = float((x - y).norm() / (x.norm() + 1e-300)) if x.shape == y.shape else float("inf")
            if r > worst:
                worst, worst_k = r, k
    R = {"what": __doc__.strip().splitlines()[0], "label": z.label,
         "host": os.uname().nodename, "board_class": z.board, "card": z.card,
         "aiclk_during": z.aiclk,
         "a": z.a, "b": z.b, "tensors_compared": len(keys),
         "only_in_a": sorted(set(A) - set(B))[:8], "only_in_b": sorted(set(B) - set(A))[:8],
         "n_only_in_a": len(set(A) - set(B)), "n_only_in_b": len(set(B) - set(A)),
         "tensors_that_moved": moved, "bit_identical": moved == 0 and len(keys) > 0,
         "worst_relative_move": worst, "worst_tensor": worst_k}
    json.dump(R, open(z.out, "w"), indent=2)
    print(json.dumps(R))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
