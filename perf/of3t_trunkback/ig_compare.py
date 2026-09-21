#!/usr/bin/env python3
"""The CHAIN error: dL/ds_in and dL/dz_in, ours against float64, at each scope.

A parameter gradient can disagree because the leaf formed it wrongly or because the cotangent
that arrived was already wrong. Over ONE block both sides are driven by the SAME captured
output cotangent, so the input gradient isolates the first mechanism from the second: it is
exactly what that block's backward hands to the block before it, and a 48-deep stack is 48 of
these composed. Reported the same way as everything else: rel_l2 with the norm ratio and the
cosine beside it, plus the magnitude/direction split of rel^2.
"""
import argparse
import json
import math
from pathlib import Path

import torch


def cmp(o, r):
    if o is None or r is None:
        return None
    o, r = o.double().flatten(), r.double().flatten()
    on, rn = float(o.norm()), float(r.norm())
    if rn == 0.0:
        return {"ref_norm": rn, "our_norm": on, "rel_l2": None}
    c = float(torch.dot(o, r)) / (on * rn) if on else 0.0
    rr = on / rn
    rel_sq = 1.0 + rr * rr - 2.0 * rr * c
    return {"rel_l2": float((o - r).norm() / rn), "norm_ratio": rr, "cos": c,
            "ref_norm": rn, "our_norm": on,
            "magnitude_share": ((rr - 1.0) ** 2) / rel_sq if rel_sq > 0 else None,
            "direction_share": (2.0 * rr * (1.0 - c)) / rel_sq if rel_sq > 0 else None,
            "best_rescaled_rel_l2": math.sqrt(max(1.0 - c * c, 0.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[], metavar="NAME=OURS:REF")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"what": __doc__.strip().splitlines()[0], "scopes": {}}
    for spec in a.pair:
        name, _, rest = spec.partition("=")
        ours_p, _, ref_p = rest.partition(":")
        o = torch.load(ours_p, map_location="cpu", weights_only=False)
        r = torch.load(ref_p, map_location="cpu", weights_only=False)
        row = {"ours": ours_p, "reference": ref_p,
               "first_block": r.get("first_block"), "blocks": r.get("blocks"),
               "ds_in": cmp(o.get("ds_in"), r.get("ds_in")),
               "dz_in": cmp(o.get("dz_in"), r.get("dz_in"))}
        out["scopes"][name] = row
        for k in ("ds_in", "dz_in"):
            v = row[k]
            if v and v.get("rel_l2") is not None:
                print(f"{name:20s} {k}: rel {v['rel_l2']:.6e}  r {v['norm_ratio']:.6f}  "
                      f"cos {v['cos']:+.6f}  |ref| {v['ref_norm']:.6e}  "
                      f"|ours| {v['our_norm']:.6e}  mag {v['magnitude_share']*100:5.1f} %  "
                      f"dir {v['direction_share']*100:5.1f} %", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
