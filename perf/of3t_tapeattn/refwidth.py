#!/usr/bin/env python3
"""Does the REFERENCE itself move between the two padded widths?

Every ratio this row reports is ours-over-the-reference at one width, compared against the same
ratio at another width. That comparison only isolates OUR width growth if the denominator is the
same object at both widths. Upstream masks its own pads, so it should be, but a should-be is not
a reading (D189: a reference is part of the measurement's identity).

Both runs carry the same 56 real tokens. This pulls those 56 rows out of each ladder by the
boundary's own single_mask, in mask order, and scores the crop-64 reference against the padded-384
one rung by rung. Anything the reference moves is subtracted from what this row may attribute to
the port.

    refwidth.py --a <ref_c64.pt> --a-boundary <boundary_c64.pt>
                --b <ref_n384.pt> --b-boundary <boundary_n384.pt> --out REFWIDTH.json
"""
from __future__ import annotations

import argparse
import json

import torch


def real_idx(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    sm = (b["single_mask"].reshape(-1) > 0)
    return torch.nonzero(sm).reshape(-1), int(sm.numel())


def pull(t, track, n, idx):
    if t is None:
        return None
    t = t.to(torch.float64)
    if track == "ds":
        return t.reshape(n, -1)[idx]
    return t.reshape(n, n, -1)[idx][:, idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-boundary", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-boundary", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ia, na = real_idx(a.a_boundary)
    ib, nb = real_idx(a.b_boundary)
    if ia.numel() != ib.numel():
        raise SystemExit(f"different real-token counts: {ia.numel()} vs {ib.numel()}")

    da = torch.load(a.a, map_location="cpu", weights_only=False)["cot"]
    db = torch.load(a.b, map_location="cpu", weights_only=False)["cot"]

    rungs = sorted(set(int(k) for k in da) & set(int(k) for k in db))
    worst = {"rel_l2": 0.0, "rung": None, "track": None}
    per_rung = {}
    for k in rungs:
        for track in ("ds", "dz"):
            x = pull(da[k].get(track), track, na, ia)
            y = pull(db[k].get(track), track, nb, ib)
            if x is None or y is None:
                continue
            nx = float(torch.linalg.vector_norm(x))
            if nx == 0.0:
                continue
            rel = float(torch.linalg.vector_norm(x - y) / nx)
            per_rung.setdefault(str(k), {})[track] = {
                "rel_l2": rel,
                "norm_ratio": float(torch.linalg.vector_norm(y)) / nx,
                "a_norm": nx,
            }
            if rel > worst["rel_l2"]:
                worst = {"rel_l2": rel, "rung": k, "track": track}

    body = {
        "what": __doc__.strip().splitlines()[0],
        "label": a.label,
        "a": a.a, "b": a.b,
        "real_tokens": int(ia.numel()),
        "padded_a": na, "padded_b": nb,
        "rungs": len(rungs),
        "worst_rel_l2_over_the_real_tokens": worst,
        "per_rung": per_rung,
    }
    json.dump(body, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in body.items() if k != "per_rung"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
