#!/usr/bin/env python3
"""The published float64 step-1 gradient, per tensor, against the bijection that can reach it.

Runs on qb2 where the bundle lives, CPU only, no device opened. It answers the question
instrument A cannot answer until D14 clears: **of the reference gradient we are eventually
going to be compared against, how much of it is currently reachable?**

`of3t-equivalence`'s bijection manifest maps 3,275 of their 4,935 tensors and lists 1,660 as
out of scope, each because the module needs a card to construct. A count is not the right
denominator for a gradient claim -- a mapped set that holds 5 % of the gradient norm would
make "3,275 of 4,935" a flattering number. So the split is taken in the gradient's own
currency: per-tensor L2 in float64, mapped against unmapped, with the largest tensors on
each side named and located.

It also re-verifies the two bundle-consumer rules that can silently void a comparison: the
presence pattern as a SET against the published `grad_presence_recycles0.json` (SS3b, `None`
is not zero), and the file's sha256 against the manifest before anything is read.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch

BUNDLE = "/home/ttuser/of3t/bundle_min"
GRADS = f"{BUNDLE}/grads_f64_recycles0.pt"
PRESENCE = f"{BUNDLE}/grad_presence_recycles0.json"
MANIFEST = f"{BUNDLE}/MANIFEST.json"
BIJECTION = sys.argv[1] if len(sys.argv) > 1 else "bijection_manifest.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "reference_profile.json"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    man = json.load(open(MANIFEST))
    declared = {a["file"]: a.get("sha256") for a in man["artifacts"]}
    got = sha256(GRADS)
    rep = {"instrument": "the published float64 gradient, per tensor, against the bijection",
           "host": os.uname().nodename, "device": "none opened -- CPU only",
           "grads_file": GRADS, "sha256": got,
           "declared_sha256": declared.get("grads_f64_recycles0.pt"),
           "sha256_matches": got == declared.get("grads_f64_recycles0.pt"),
           "torch": torch.__version__}
    if not rep["sha256_matches"]:
        print("sha256 mismatch on the reference gradient; refusing to profile")
        json.dump(rep, open(OUT, "w"), indent=1)
        return 1

    grads = torch.load(GRADS, map_location="cpu", weights_only=False)
    presence = json.load(open(PRESENCE))
    bij = json.load(open(BIJECTION))["manifest"]

    norms, absent, zero = {}, [], []
    for name, g in grads.items():
        if g is None:
            absent.append(name)
            continue
        n = float(torch.linalg.vector_norm(g.to(torch.float64)).item())
        norms[name] = n
        if n == 0.0:
            zero.append(name)

    # SS3b: the presence pattern as a set, before any magnitude.
    pres_true = {k for k, v in presence.items()
                 if (v if isinstance(v, bool) else v.get("has_grad", v.get("present")))}
    rep["presence"] = {
        "tensors_in_file": len(grads), "with_a_gradient": len(norms),
        "absent": len(absent), "exactly_zero": len(zero),
        "published_presence_true": len(pres_true),
        "set_agrees": sorted(norms) == sorted(pres_true),
        "in_file_not_in_published": sorted(set(norms) - pres_true)[:10],
        "in_published_not_in_file": sorted(pres_true - set(norms))[:10]}

    mapped = {n: v for n, v in norms.items() if n in bij}
    unmapped = {n: v for n, v in norms.items() if n not in bij}
    tot = float(np.linalg.norm(list(norms.values())))
    nm = float(np.linalg.norm(list(mapped.values()) or [0.0]))
    nu = float(np.linalg.norm(list(unmapped.values()) or [0.0]))
    top = lambda d, k=10: [{"tensor": t, "l2": v} for t, v in
                           sorted(d.items(), key=lambda kv: -kv[1])[:k]]
    by_prefix = {}
    for n, v in unmapped.items():
        by_prefix.setdefault(n.split(".")[0], [0, 0.0])
        by_prefix[n.split(".")[0]][0] += 1
        by_prefix[n.split(".")[0]][1] += v * v
    rep["bijection_split"] = {
        "their_tensors_with_a_gradient": len(norms),
        "mapped_by_the_bijection": len(mapped),
        "unmapped": len(unmapped),
        "total_l2_over_all_tensors": tot,
        "l2_in_mapped": nm, "l2_in_unmapped": nu,
        "fraction_of_gradient_norm_reachable": (nm * nm) / (tot * tot) if tot else 0.0,
        "largest_mapped": top(mapped),
        "largest_unmapped": top(unmapped),
        "unmapped_by_top_level": {k: {"tensors": c, "l2": float(np.sqrt(s))}
                                  for k, (c, s) in sorted(by_prefix.items())},
        "note": "the denominator a gradient claim owes is the gradient, not the tensor "
                "count: these two numbers are what makes 3275 of 4935 readable"}

    w = max(norms.items(), key=lambda kv: kv[1])
    rep["largest_tensor"] = {"tensor": w[0], "l2": w[1],
                             "mapped": w[0] in bij,
                             "shape": list(grads[w[0]].shape)}
    json.dump(rep, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"tensors {len(grads)}, with a gradient {len(norms)}, absent {len(absent)}, "
          f"exactly zero {len(zero)}; presence set agrees: {rep['presence']['set_agrees']}")
    print(f"mapped {len(mapped)} / {len(norms)} holding "
          f"{rep['bijection_split']['fraction_of_gradient_norm_reachable']:.4f} of the "
          f"squared gradient norm; total L2 {tot:.6f}")
    print(f"largest tensor overall: {w[0]} at L2 {w[1]:.6f} "
          f"(mapped: {w[0] in bij})")
    print(f"largest unmapped: {rep['bijection_split']['largest_unmapped'][0]}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
