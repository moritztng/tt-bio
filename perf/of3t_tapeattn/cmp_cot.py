#!/usr/bin/env python3
"""Compare two cotangent ladders tensor by tensor.

The whole .pt file carries the arm's report beside the tensors: paths, lever names, counters.
So a file sha256 answers "same run", not "same numbers". This hashes only the PAYLOAD, in rung
order, and reports the worst relative L2 over all rungs beside it. That is the reading the
instrument-inertness proof and the A/A floor both need.

    cmp_cot.py <a.pt> <b.pt> --label A384-vs-C384 --out CMP.json
"""
from __future__ import annotations

import argparse
import hashlib
import json

import torch


def content_sha(cot) -> str:
    h = hashlib.sha256()
    for rung in sorted(cot, key=lambda r: int(r)):
        for track in ("ds", "dz"):
            t = cot[rung].get(track)
            if t is None:
                h.update(b"none")
                continue
            t = t.contiguous()
            h.update(str(rung).encode())
            h.update(track.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(str(t.dtype).encode())
            h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    da = torch.load(a.a, map_location="cpu", weights_only=False)
    db = torch.load(a.b, map_location="cpu", weights_only=False)
    ca, cb = da["cot"], db["cot"]

    rungs = sorted(set(ca) & set(cb), key=lambda r: int(r))
    worst = {"rel_l2": 0.0, "rung": None, "track": None}
    bitexact = 0
    total = 0
    per_rung = {}
    for r in rungs:
        for track in ("ds", "dz"):
            ta, tb = ca[r].get(track), cb[r].get(track)
            if ta is None or tb is None:
                continue
            total += 1
            if torch.equal(ta, tb):
                bitexact += 1
                rel = 0.0
            else:
                x, y = ta.double(), tb.double()
                n = torch.linalg.vector_norm(x).item()
                rel = (torch.linalg.vector_norm(x - y).item() / n) if n else float("inf")
            per_rung.setdefault(str(r), {})[track] = rel
            if rel > worst["rel_l2"]:
                worst = {"rel_l2": rel, "rung": int(r), "track": track}

    body = {
        "label": a.label,
        "a": a.a, "b": a.b,
        "rungs_compared": len(rungs),
        "tensors_compared": total,
        "tensors_bit_identical": bitexact,
        "content_sha256_a": content_sha(ca),
        "content_sha256_b": content_sha(cb),
        "worst_rel_l2": worst,
        "per_rung_rel_l2": per_rung,
    }
    body["identical"] = (body["content_sha256_a"] == body["content_sha256_b"])
    json.dump(body, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in body.items() if k != "per_rung_rel_l2"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
