#!/usr/bin/env python3
"""Cut the trunk boundary out of the captured step once, so every arm reads the same bytes.

`of3t_gradients/cap` holds three 482 MB block boundaries of upstream's own float64 forward of the
frozen step with dropout replayed at r = 0. Every arm in this row is driven from block 0's input
and read at block 47's output. Slicing that out to one small file makes the boundary an artifact
with a sha256 (A24) instead of a slicing convention each script re-derives -- which is how
`of3t-trunkfwd` ended up asserting bit-identity between its own arms by hand.

Writes s_in, z_in, single_mask, pair_mask and the captured block-47 output, all float64, plus the
padding fraction (D95) as a number rather than a remark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

CAP = "/home/ttuser/of3t_gradients/cap"


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default=CAP)
    ap.add_argument("--crop", type=int, default=64, help="0 means the whole 384-token batch")
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    last = a.blocks - 1
    src0 = os.path.join(a.cap, "block0_boundary.pt")
    srcL = os.path.join(a.cap, f"block{last}_boundary.pt")
    cap0 = torch.load(src0, map_location="cpu", weights_only=False)
    capL = torch.load(srcL, map_location="cpu", weights_only=False)
    c = a.crop

    cs = (lambda x: x[:, :c].contiguous()) if c else (lambda x: x.contiguous())
    cz = (lambda x: x[:, :c, :c].contiguous()) if c else (lambda x: x.contiguous())
    f = lambda x: x.detach().to(torch.float64)

    b = {"s_in": f(cs(cap0["args"][0])), "z_in": f(cz(cap0["args"][1])),
         "single_mask": f(cs(cap0["kwargs"]["single_mask"])),
         "pair_mask": f(cz(cap0["kwargs"]["pair_mask"])),
         "s_ref_050_captured": f(cs(capL["out"][0])),
         "z_ref_050_captured": f(cz(capL["out"][1])),
         "crop": c, "blocks": a.blocks,
         "provenance": {"block0": src0, "block47": srcL}}
    torch.save(b, a.out)

    sm, pm = b["single_mask"], b["pair_mask"]
    n = int(sm.shape[1])
    real = int(sm.sum())
    rep = {"what": __doc__.strip().splitlines()[0], "crop": c, "blocks": a.blocks,
           "tokens": n, "real_tokens": real,
           "padding_fraction_single": 1.0 - real / n,
           "real_over_total_single": f"{real} of {n}",
           "pair_real_cells": int(pm.sum()),
           "pair_total_cells": n * n,
           "real_over_total_pair": f"{int(pm.sum())} of {n * n}",
           "padding_fraction_pair": 1.0 - float(pm.sum()) / (n * n),
           "norms": {k: float(b[k].norm()) for k in
                     ("s_in", "z_in", "s_ref_050_captured", "z_ref_050_captured")},
           "source_sha256": {src0: sha256_file(src0), srcL: sha256_file(srcL)},
           "out": a.out, "out_sha256": sha256_file(a.out)}
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({k: rep[k] for k in
                      ("tokens", "real_tokens", "real_over_total_single",
                       "real_over_total_pair", "out_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
