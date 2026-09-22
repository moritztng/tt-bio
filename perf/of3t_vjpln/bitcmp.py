#!/usr/bin/env python3
"""Two cotangent ladders, compared BITWISE. The inertness controls need this and not `rel_l2`.

A control that round-trips every selected node through host float64 and writes the SAME value
back has to come out byte for byte identical. `rel_l2 == 0.0` is the same statement only if the
comparison is done in a wider type than the store, so this counts exactly-equal elements
instead and reports the first tensor that is not.
"""
from __future__ import annotations

import hashlib
import json
import sys

import torch

A = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["cot"]
B = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["cot"]
out = {"a": sys.argv[1], "b": sys.argv[2], "tensors": 0, "bit_identical": 0, "differing": []}
h = hashlib.sha256()
for k in sorted(int(x) for x in A):
    for tr in ("ds", "dz"):
        ta, tb = A[k].get(tr), B.get(k, {}).get(tr)
        if ta is None and tb is None:
            continue
        out["tensors"] += 1
        h.update(ta.contiguous().numpy().tobytes())
        if tb is not None and ta.shape == tb.shape and torch.equal(ta, tb):
            out["bit_identical"] += 1
        else:
            d = (ta.double() - tb.double()) if tb is not None else None
            out["differing"].append(
                {"rung": k, "track": tr,
                 "n_differing": (int((ta != tb).sum()) if tb is not None else None),
                 "max_abs": (float(d.abs().max()) if d is not None else None),
                 "rel_l2": (float(torch.linalg.vector_norm(d.reshape(-1))
                                  / torch.linalg.vector_norm(ta.double().reshape(-1)))
                            if d is not None else None)})
out["a_content_sha256_12"] = h.hexdigest()[:12]
out["verdict"] = ("BIT-IDENTICAL" if out["bit_identical"] == out["tensors"]
                  else "DIFFERS on %d of %d" % (out["tensors"] - out["bit_identical"],
                                                out["tensors"]))
print(json.dumps({k: v for k, v in out.items() if k != "differing"}, indent=1))
for r in out["differing"][:6]:
    print(" ", r)
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    print("wrote", sys.argv[3])
