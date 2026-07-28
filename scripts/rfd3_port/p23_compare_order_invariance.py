"""Acceptance report for p23: bit-exactness, not a PCC threshold."""
import sys
from pathlib import Path

import torch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p23/verify")


def load(tag):
    p = OUT / f"{tag}.pt"
    return torch.load(p, weights_only=False) if p.exists() else {}


ref = {}
for tag in ("ref_40", "ref_150", "ref_250", "ref_mpro"):
    ref.update(load(tag))
fix_iso = {}
for tag in ("fix_40", "fix_150", "fix_250", "fix_mpro"):
    fix_iso.update(load(tag))

fail = 0


def cmp(name, a, b):
    global fail
    keys = sorted(set(a) & set(b))
    if not keys:
        print(f"  {name}: NO OVERLAP")
        fail += 1
        return
    worst = 0.0
    for k in keys:
        m = float((a[k] - b[k]).abs().max())
        worst = max(worst, m)
        flag = "" if m == 0.0 else "   <-- NOT BIT-EXACT"
        print(f"  {name:22s} {k:28s} maxabs={m:.6e}{flag}")
        if m != 0.0:
            fail += 1
    return worst


print("=== fix vs pre-fix, each design folded ALONE (fix must change nothing) ===")
cmp("fix_iso vs ref_iso", fix_iso, ref)
print("=== fixed tree: sequenced vs isolated, four orderings (must all be 0.0) ===")
for tag in ("seq_asc", "seq_desc", "seq_mid", "seq_big1"):
    d = load(tag)
    if d:
        cmp(f"{tag} vs fix_iso", d, fix_iso)
print("=== control: PRE-FIX tree sequenced vs isolated (expected to be broken) ===")
pre = load("pre_asc")
if pre:
    for k in sorted(set(pre) & set(ref)):
        m = float((pre[k] - ref[k]).abs().max())
        print(f"  pre_asc vs ref_iso     {k:28s} maxabs={m:.6e}")
print()
print("VERDICT:", "BIT-EXACT (all cells 0.0)" if fail == 0 else f"{fail} cell(s) NOT bit-exact")
