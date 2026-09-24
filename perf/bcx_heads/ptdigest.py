#!/usr/bin/env python3
"""Per-tensor sha256 of a gradient .pt, so two arms can be compared bit for bit without both
1.3 GB files on one disk, or on one host.

    ptdigest.py <arm.pt> <out.json>          digest one file
    ptdigest.py --cmp <a.json> <b.json>      count the tensors that are bit-identical

Digest = sha256 over dtype, shape and the raw bytes. CPU only.
"""
import hashlib
import json
import sys

import torch


def grads(d):
    for k in ("grads", "gradients", "ours", "g"):         # pairdiff.py's _grads, verbatim order
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]
    return d if isinstance(d, dict) else {}


def digest(path):
    g = grads(torch.load(path, map_location="cpu"))
    out = {}
    for k, t in g.items():
        if torch.is_tensor(t):
            t = t.detach().contiguous()
            h = hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
            out[k] = h.hexdigest()
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--cmp":
        a, b = (json.load(open(p))["tensors"] for p in sys.argv[2:4])
        common = sorted(set(a) & set(b))
        same = sum(a[k] == b[k] for k in common)
        print(json.dumps({"a": sys.argv[2], "b": sys.argv[3], "a_n": len(a), "b_n": len(b),
                          "common": len(common), "bit_identical": same,
                          "differ": [k for k in common if a[k] != b[k]][:20]}))
    else:
        d = digest(sys.argv[1])
        json.dump({"pt": sys.argv[1], "n": len(d), "tensors": d}, open(sys.argv[2], "w"), indent=0)
        print(sys.argv[1], len(d), "tensors")
