#!/usr/bin/env python3
"""Per-tensor reference gradient mass, so a coverage share can be taken in the campaign's own
denominator without a 2.9 GB load every time.

Reads the bundle's float64 reference gradient (`grads_f64_043.pt`, 4,170 tensors) and writes
name -> squared float64 norm. The total must reproduce `SECTION_MASS_MEASURED.json`'s
10.27964, which is the check that this is the same denominator the 92.1568 % is taken in and
not a second one that happens to be nearby.
"""
from __future__ import annotations

import json
import sys

import torch

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "perf/of3t_pathcov/REF_MASS.json"

try:
    d = torch.load(SRC, map_location="cpu", weights_only=False, mmap=True)
except Exception:
    d = torch.load(SRC, map_location="cpu", weights_only=False)
if isinstance(d, dict) and "grads" in d and isinstance(d["grads"], dict):
    d = d["grads"]

mass, total, skipped = {}, 0.0, []
for k, v in d.items():
    if not torch.is_tensor(v):
        skipped.append(k)
        continue
    m = float(v.double().pow(2).sum())
    mass[k] = m
    total += m
json.dump({
    "source": SRC,
    "n_tensors": len(mass),
    "total_sq_norm": total,
    "expected_total": 10.27964,
    "skipped_non_tensor": skipped[:20],
    "mass": mass,
}, open(OUT, "w"), indent=1)
print(f"{len(mass)} tensors, total sq norm {total!r} (expected 10.27964), wrote {OUT}")
