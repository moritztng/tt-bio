#!/usr/bin/env python3
"""Two of OUR cotangent ladders against each other, masked to the real tokens.

The pad-zero control needs this and not `cmp_cot.py`: the question is not whether two runs are
byte-identical, it is how far apart they are ON THE 56 REAL TOKENS, which is the only place the
figures this campaign quotes are read.

No new reference is needed for it. `of3t-tapeattn` established that the in-frame float64
reference is bit-identical between padded 64 and padded 384 on the real tokens, rel_l2 exactly
0.0 at all 49 rungs on both tracks, with 8 pad rows against 328. Exact arithmetic therefore does
not see a pad cell at all, so any real-token difference a pad change makes is OURS.
"""
from __future__ import annotations

import argparse
import json

import torch


def triple(m, r):
    m = m.reshape(-1).to(torch.float64)
    r = r.reshape(-1).to(torch.float64)
    nr = float(torch.linalg.vector_norm(r))
    nm = float(torch.linalg.vector_norm(m))
    if nr == 0.0:
        return None
    return {"rel_l2": float(torch.linalg.vector_norm(m - r) / nr),
            "norm_ratio": nm / nr,
            "cos": (float((m @ r) / (nm * nr)) if nm else 0.0),
            "ref_norm": nr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="the baseline ladder")
    ap.add_argument("b", help="the ladder under test")
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    o = ap.parse_args()

    bd = torch.load(o.boundary, map_location="cpu", weights_only=False)
    sm = (bd["single_mask"].reshape(-1) > 0).to(torch.float64)
    N = int(sm.numel())
    real = int(sm.sum())
    mds = sm.reshape(N, 1)
    mdz = (sm.reshape(N, 1) * sm.reshape(1, N)).reshape(N, N, 1)

    A = torch.load(o.a, map_location="cpu", weights_only=False)["cot"]
    B = torch.load(o.b, map_location="cpu", weights_only=False)["cot"]

    out = {"label": o.label, "a": o.a, "b": o.b,
           "masking": f"{real} real of {N} tokens", "rungs": {}}
    worst = {"rel_l2": 0.0, "rung": None, "track": None}
    for k in sorted(int(x) for x in A):
        row = {}
        for track, msk in (("ds", mds), ("dz", mdz)):
            ta, tb = A[k].get(track), B.get(k, {}).get(track)
            if ta is None or tb is None:
                continue
            shp = (N, -1) if track == "ds" else (N, N, -1)
            t = triple(tb.to(torch.float64).reshape(shp) * msk,
                       ta.to(torch.float64).reshape(shp) * msk)
            row[track] = t
            if t and t["rel_l2"] > worst["rel_l2"]:
                worst = {"rel_l2": t["rel_l2"], "rung": k, "track": track}
        out["rungs"][k] = row
    out["worst_masked_rel_l2"] = worst
    json.dump(out, open(o.out, "w"), indent=1)

    print(f"{'rung':>4} {'ds rel':>12} {'ds nr':>9} {'ds cos':>10} "
          f"{'dz rel':>12} {'dz nr':>9} {'dz cos':>10}")
    for k in sorted(out["rungs"], reverse=True):
        r = out["rungs"][k]
        def c(tr, f):
            e = r.get(tr)
            return float("nan") if not e else e[f]
        print(f"{k:>4} {c('ds','rel_l2'):12.5e} {c('ds','norm_ratio'):9.5f} "
              f"{c('ds','cos'):10.6f} {c('dz','rel_l2'):12.5e} "
              f"{c('dz','norm_ratio'):9.5f} {c('dz','cos'):10.6f}")
    print(json.dumps(worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
