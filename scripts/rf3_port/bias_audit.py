#!/usr/bin/env python3
"""Audit which RF3 projections carry a bias that tt-bio's shared blocks do not read.

Twice now a tt-bio block has silently dropped a non-zero RF3 bias: TriangleAttention's
gate and output projections (192 vectors, |max| up to 1.32, found only because the
remap coverage check reports leftovers), and OuterProductMean's left/right
projections. The AF3-lineage models tt-bio already supports -- Boltz-2, Protenix-v2,
OpenFold3 -- leave these bias-free, so the shared blocks never needed them.

Rather than find the rest one PCC disappointment at a time, this lists every biased
projection in the RF3 checkpoint, grouped by the module that owns it, with the
magnitude. Anything here whose tt-bio counterpart has no bias needs optional bias
support before that block can be reused.

    python scripts/rf3_port/bias_audit.py --ckpt /path/to/rf3_latest.ckpt
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import torch

PREFIX = "shadow."

#: Leaf names that are a projection bias rather than a norm/embedding bias. Norm
#: biases are always read; it is the linear-projection ones that get dropped.
NORM_HINTS = ("norm", "layer_norm", "ln_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--min-absmax", type=float, default=0.0,
                    help="only report biases above this magnitude")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]

    groups: dict[str, list] = defaultdict(list)
    for key, value in sd.items():
        if not key.startswith(PREFIX) or not key.endswith(".bias"):
            continue
        rel = key[len(PREFIX):]
        leaf = rel.rsplit(".", 2)[-2]           # the module owning the bias
        if any(h in leaf for h in NORM_HINTS):
            continue                             # norm biases are never dropped
        absmax = float(value.abs().max())
        if absmax < args.min_absmax:
            continue
        # collapse repeated stack indices so 48 blocks report as one row
        owner = re.sub(r"\.\d+\.", ".*.", rel.rsplit(".", 1)[0])
        groups[owner].append(absmax)

    rows = []
    for owner, mags in sorted(groups.items()):
        rows.append({
            "projection": owner,
            "instances": len(mags),
            "absmax_max": round(max(mags), 4),
            "absmax_median": round(sorted(mags)[len(mags) // 2], 4),
            "all_nonzero": all(m > 0 for m in mags),
        })

    print(json.dumps({"biased_projections": len(rows), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
