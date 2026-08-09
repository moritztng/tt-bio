#!/usr/bin/env python3
"""Score the C2FIX gate: bit-exactness against BASE, and the fold ratio.

Two timing views, because the box is shared and a sibling job on another card moves a fold by
more than the effect being measured. The sweep view is one arm after the other; the paired view
alternates the arms round by round and pools every warm fold, which is the one to trust.

    python3 perf/w6_c2fix/score.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"


def _load(pat: str) -> list[dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(str(OUT / pat)))]


def main() -> int:
    print("=== sweep (arms back to back, --repeat 5 at 298 / 3 at 117) ===")
    rows = [d for d in _load("*.json")
            if not re.search(r"_r\d+\.json$", str(d)) and "arm" in d]
    rows = [json.load(open(f)) for f in sorted(glob.glob(str(OUT / "*.json")))
            if re.match(r"^(BASE|C2FIX)_[a-z0-9-]+_\d+\.json$", os.path.basename(f))]
    by = {(d["arm"], d["model"], d["size"]): d for d in rows}
    for size in ("298", "117"):
        for m in ("protenix-v2", "opendde"):
            b, c = by.get(("BASE", m, size)), by.get(("C2FIX", m, size))
            if not (b and c):
                continue
            print(f"  {m:12s} {size:>3s}  BASE med {b['warm_median_s']:7.3f} min {b['warm_min_s']:7.3f}"
                  f"   C2FIX med {c['warm_median_s']:7.3f} min {c['warm_min_s']:7.3f}"
                  f"   ratio med {b['warm_median_s']/c['warm_median_s']:.3f}x"
                  f" min {b['warm_min_s']/c['warm_min_s']:.3f}x")
            print(f"      warm BASE  {b['warm_s']}")
            print(f"      warm C2FIX {c['warm_s']}")

    print("\n=== paired (arms alternating, all warm folds pooled) ===")
    for m in ("protenix-v2", "opendde"):
        pool = {}
        for arm in ("BASE", "C2FIX"):
            ws = []
            for f in sorted(glob.glob(str(OUT / f"{arm}_{m}_298_r*.json"))):
                ws += json.load(open(f))["warm_s"]
            pool[arm] = sorted(ws)
        if not pool["BASE"] or not pool["C2FIX"]:
            continue
        b, c = pool["BASE"], pool["C2FIX"]
        bm, cm = b[len(b) // 2], c[len(c) // 2]
        print(f"  {m:12s} 298  n={len(b)}/{len(c)}  BASE med {bm:7.3f} min {b[0]:7.3f}"
              f"   C2FIX med {cm:7.3f} min {c[0]:7.3f}"
              f"   ratio med {bm/cm:.3f}x min {b[0]/c[0]:.3f}x")
        print(f"      warm BASE  {[round(x, 2) for x in b]}")
        print(f"      warm C2FIX {[round(x, 2) for x in c]}")

    print("\n=== bit-exactness (rank-0 coordinates, full float precision) ===")
    for size in ("298", "117"):
        for m in ("protenix-v2", "opendde"):
            a = OUT / f"BASE_{m}_{size}" / "coords.npy"
            z = OUT / f"C2FIX_{m}_{size}" / "coords.npy"
            if not (a.is_file() and z.is_file()):
                continue
            A, B = np.load(a), np.load(z)
            print(f"  {m:12s} {size:>3s}  shape {A.shape}  array_equal={np.array_equal(A, B)}"
                  f"  max abs delta {np.abs(A - B).max():.3e} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
