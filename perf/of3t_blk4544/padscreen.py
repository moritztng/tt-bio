#!/usr/bin/env python3
"""The null candidate, screened with no device time.

At padded 384 there are 328 pad tokens against 8 at padded 64, and every reduction over the
token axis runs over them. If OUR cotangent carries mass on the pad rows that the float64
reference's does not, a masked reading can step without any arithmetic getting worse.

Reads the cotangent files `of3t-tapeattn` already wrote plus its float64 reference, and reports
per rung, per track: the pad-row share of the squared norm, ours and the reference's.
"""
from __future__ import annotations

import argparse
import json

import torch


def shares(t, msk_flat, track, N):
    if t is None:
        return None
    t = t.to(torch.float64)
    t = t.reshape(N, -1) if track == "ds" else t.reshape(N, N, -1)
    tot = float((t * t).sum())
    if tot == 0.0:
        return {"sq_norm": 0.0, "pad_share": None}
    if track == "ds":
        real = float(((t * msk_flat.reshape(N, 1)) ** 2).sum())
    else:
        m2 = (msk_flat.reshape(N, 1) * msk_flat.reshape(1, N)).reshape(N, N, 1)
        real = float(((t * m2) ** 2).sum())
    return {"sq_norm": tot, "pad_share": (tot - real) / tot}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref-f64", required=True)
    ap.add_argument("--width", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    sm = (b["single_mask"].reshape(-1) > 0).to(torch.float64)
    N = int(sm.numel())
    real = int(sm.sum())

    ours = torch.load(a.ours, map_location="cpu", weights_only=False)["cot"]
    ref = torch.load(a.ref_f64, map_location="cpu", weights_only=False)["cot"]

    out = {"width": a.width, "tokens": {"real": real, "padded": N},
           "what": "pad-row share of the squared norm, ours and the in-frame float64 reference",
           "rungs": {}}
    for k in sorted(int(x) for x in ours):
        row = {}
        for track in ("ds", "dz"):
            o = shares(ours[k].get(track), sm, track, N)
            r = shares(ref.get(k, {}).get(track), sm, track, N)
            if o is None and r is None:
                continue
            row[track] = {"ours": o, "ref": r}
        out["rungs"][k] = row
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"{'rung':>4} {'ds pad ours':>12} {'ds pad ref':>12} "
          f"{'dz pad ours':>12} {'dz pad ref':>12}")
    for k in sorted(out["rungs"], reverse=True):
        r = out["rungs"][k]
        def c(tr, who):
            e = r.get(tr, {}).get(who)
            v = e and e.get("pad_share")
            return float("nan") if v is None else v
        print(f"{k:>4} {c('ds','ours'):12.4e} {c('ds','ref'):12.4e} "
              f"{c('dz','ours'):12.4e} {c('dz','ref'):12.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
