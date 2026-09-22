#!/usr/bin/env python3
"""of3t-padshape deliverable 2: which sub-module, and which block, carries the width dependence.

No extra device arm. The sweep's own tensors already hold every parameter gradient of all 48
blocks at every width, so the bisection is a grouping of tensors that exist rather than a
one-block probe of tensors that do not. Full depth, the real stack, the real cotangent.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

O = Path("/tmp/of3t/of3t-padshape")
ARMS = {
    64: O / "dev_CTRL_w64_A.pt", 128: O / "dev_CTRL_w128.pt", 192: O / "dev_CTRL_w192.pt",
    256: O / "dev_CTRL_w256.pt",
    384: Path("/tmp/of3t/of3t-modelboundary/dev_CTRL_n384_nocaptures.pt"),
}
RENORM = {64: O / "dev_RENORM_w64.pt", 256: O / "dev_RENORM_w256.pt",
          384: Path("/tmp/of3t/of3t-modelboundary/dev_RENORM_n384_nocaptures.pt")}
BLK = re.compile(r"^pairformer_stack\.blocks\.(\d+)\.([^.]+)\.")


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = {k: v.to(torch.float64).reshape(-1) for k, v in d["grads"].items() if v is not None}
    del d
    return g


def group(g):
    sub, blk, subblk = defaultdict(float), defaultdict(float), defaultdict(float)
    for k, v in g.items():
        m = BLK.match(k)
        if not m:
            sub["(unmatched)"] += float(v @ v)
            continue
        j, s = int(m.group(1)), m.group(2)
        q = float(v @ v)
        sub[s] += q
        blk[j] += q
        subblk[(s, j)] += q
    return sub, blk, subblk


out = {"what": __doc__.strip().splitlines()[0], "host": "tt-quietbox2 (qb2), card 0, p300c",
       "arms": {}, "renorm_arms": {}}
G, S, B, SB = {}, {}, {}, {}
for w, p in ARMS.items():
    if not p.exists():
        print(f"MISSING w{w}: {p}")
        continue
    G[w] = load(p)
    S[w], B[w], SB[w] = group(G[w])
    out["arms"][w] = {"path": str(p), "n": len(G[w])}
    print(f"w{w}: {len(G[w])} tensors, {len(S[w])} sub-modules", flush=True)

widths = sorted(S)
lo, hi = widths[0], widths[-1]
subs = sorted(S[hi], key=lambda s: -S[hi][s])
tot = {w: sum(S[w].values()) ** 0.5 for w in widths}
out["total_grad_norm"] = {w: tot[w] for w in widths}
out["per_submodule"] = {}
print(f"\n{'sub-module':28s} " + " ".join(f"{w:>12d}" for w in widths)
      + f"   ratio {hi}/{lo}   share@{hi}")
for s in subs:
    n = {w: S[w].get(s, 0.0) ** 0.5 for w in widths}
    r = n[hi] / n[lo] if n[lo] > 0 else float("inf")
    share = 100.0 * S[hi].get(s, 0.0) / sum(S[hi].values())
    out["per_submodule"][s] = {"norm": n, "ratio_hi_over_lo": r, "pct_of_sq_norm_at_hi": share}
    print(f"{s:28s} " + " ".join(f"{n[w]:12.6f}" for w in widths) + f"   {r:10.4f}   {share:7.3f} %")

out["per_block"] = {}
print(f"\nblock   " + " ".join(f"{w:>12d}" for w in widths) + f"   ratio {hi}/{lo}")
for j in sorted(B[hi]):
    n = {w: B[w].get(j, 0.0) ** 0.5 for w in widths}
    r = n[hi] / n[lo] if n[lo] > 0 else float("inf")
    out["per_block"][j] = {"norm": n, "ratio_hi_over_lo": r}
    if j % 6 == 0 or j in (1, 46, 47):
        print(f"{j:5d}   " + " ".join(f"{n[w]:12.6f}" for w in widths) + f"   {r:10.4f}")

# the worst single tensor by width growth, among tensors that carry real mass at w=lo
cand = [(float((G[hi][k] @ G[hi][k]) ** 0.5 / max((G[lo][k] @ G[lo][k]) ** 0.5, 1e-30)), k)
        for k in G[hi] if k in G[lo] and float(G[lo][k] @ G[lo][k]) ** 0.5 > 1e-6]
cand.sort(reverse=True)
out["worst_growth_tensors"] = [{"ratio": r, "param": k} for r, k in cand[:8]]
print("\nworst width growth, per tensor:")
for r, k in cand[:8]:
    print(f"  {r:12.4f}  {k}")

# the same grouping on the shipped RENORM=1 arm, so the law can be read on both conventions
RG = {}
for w, p in RENORM.items():
    if p.exists():
        RG[w] = group(load(p))[0]
        out["renorm_arms"][w] = {"path": str(p)}
if len(RG) >= 2:
    rw = sorted(RG)
    rlo, rhi = rw[0], rw[-1]
    out["per_submodule_renorm"] = {}
    print(f"\nRENORM=1  {'sub-module':26s} " + " ".join(f"{w:>12d}" for w in rw)
          + f"   ratio {rhi}/{rlo}")
    for s in sorted(RG[rhi], key=lambda s: -RG[rhi][s]):
        n = {w: RG[w].get(s, 0.0) ** 0.5 for w in rw}
        r = n[rhi] / n[rlo] if n[rlo] > 0 else float("inf")
        out["per_submodule_renorm"][s] = {"norm": n, "ratio_hi_over_lo": r}
        print(f"          {s:26s} " + " ".join(f"{n[w]:12.6f}" for w in rw) + f"   {r:10.4f}")
    out["renorm_total_grad_norm"] = {w: sum(RG[w].values()) ** 0.5 for w in rw}

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("\nwrote " + sys.argv[1])
