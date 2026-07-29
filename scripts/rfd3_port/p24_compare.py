"""Compare p24's attn_indices rewrite against the tree at HEAD, cell by cell.

The rewrite changes allocation, not arithmetic -- one design at a time instead of a batched
(D,L,L), and masked_fill_ instead of where() -- so the bar is bit-exactness, not a tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p24/verify")
TAGS = ("40", "250", "mpro")

fail = 0
worst = 0.0
print("=== patched tree vs HEAD, every design folded alone ===")
for tag in TAGS:
    ref_p = OUT / f"ref_{tag}.pt"
    # the runner's `run` helper shadows $tag, so some patched-tree files landed as
    # fix_ref_<tag>.pt rather than fix_<tag>.pt -- same data, take whichever exists
    fix_p = next((p for p in (OUT / f"fix_{tag}.pt", OUT / f"fix_ref_{tag}.pt")
                  if p.exists()), OUT / f"fix_{tag}.pt")
    if not (ref_p.exists() and fix_p.exists()):
        print(f"  {tag}: MISSING (ref={ref_p.exists()} fix={fix_p.exists()})")
        fail += 1
        continue
    print(f"  [{tag}] {ref_p.name} vs {fix_p.name}")
    ref, fix = torch.load(ref_p, weights_only=False), torch.load(fix_p, weights_only=False)
    keys = sorted(set(ref) & set(fix))
    if not keys:
        print(f"  {tag}: NO OVERLAP  ref={sorted(ref)[:3]} fix={sorted(fix)[:3]}")
        fail += 1
        continue
    for k in keys:
        m = float((ref[k].float() - fix[k].float()).abs().max())
        worst = max(worst, m)
        flag = "" if m == 0.0 else "   <-- NOT BIT-EXACT"
        print(f"  {tag:6s} {str(k):34s} maxabs={m:.6e}{flag}")
        if m != 0.0:
            fail += 1

print()
print("VERDICT:", "BIT-EXACT (all cells 0.0)" if fail == 0
      else f"{fail} cell(s) NOT bit-exact, worst {worst:.6e}")
raise SystemExit(1 if fail else 0)
