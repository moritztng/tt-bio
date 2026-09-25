#!/usr/bin/env python3
"""Where mass_at_or_better_than_bf16 is lost: F64 mass of tensors whose rel exceeds bf16's,
by section, with our and bf16's rel on that section's losing tensors.  missmass.py --k 3"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_fullstep64"))
from score import section_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()
d = torch.load(f"/home/ttuser/of3t_sigma/scored_s{a.k}.pt", weights_only=False)
g, o, b = d["f64"], d["ours"], d["bf16"]
m = {k: float((v * v).sum()) for k, v in g.items()}
tot = sum(m.values())
sec = defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
lost = []
for k in g:
    ro = float(((o[k] - g[k]) ** 2).sum()) ** .5 / m[k] ** .5
    rb = float(((b[k] - g[k]) ** 2).sum()) ** .5 / m[k] ** .5
    s = sec[section_of(k)]
    s[0] += m[k]
    if ro > rb:
        s[1] += m[k]; s[2] += 1; s[3] += ro * m[k]; s[4] += rb * m[k]
        lost.append((m[k] / tot, k, ro, rb))
print(f"k={a.k} total lost {sum(x[0] for x in lost):.4f}")
for n, (sm, lm, c, ro, rb) in sorted(sec.items(), key=lambda x: -x[1][1]):
    if lm:
        print(f"{n:45s} sec_mass {sm/tot:.4f} lost {lm/tot:.4f} n {c:4d}  mass-wtd rel ours {ro/lm:.3f} bf16 {rb/lm:.3f}")
for f, k, ro, rb in sorted(lost, reverse=True)[:a.top]:
    print(f"  {f:.4f} {k:70s} ours {ro:.3f} bf16 {rb:.3f}")
