#!/usr/bin/env python3
"""Fold cost before/after a split, normalised by target size.

Raw median wall_s per generator is confounded: fold cost scales with token count and the campaign
walks each slice in a fixed order, so folds after a split are not a random sample of sizes. A raw
"+71%" can be entirely target size. Dividing by the target's token count removes that -- the state
doc already records s/token as the stable quantity (2.7-3.4 across the top end).

Reports s/token per generator either side of the split, plus a sign summary across cells, because
with 1-3 folds per cell the direction across cells is worth more than any single ratio.

    abag_xm_fold_rate2.py <split_record_count>
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit("usage: abag_xm_fold_rate2.py <split_record_count>")
CUT = int(sys.argv[1])

WT = Path("/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p4")
SLICES = WT / "docs" / "implementation-parity-data" / "abag-xm-tier-a-slices.json"
tokens = json.load(open(SLICES))["tokens"]
P = Path.home() / "abag_xm" / "tier_a" / "progress.jsonl"
recs = [json.loads(l) for l in open(P) if l.strip()]


def cells(subset):
    out = defaultdict(list)
    for r in subset:
        if r.get("status") != "ok" or not r.get("wall_s"):
            continue
        tok = tokens.get(r["target"])
        if not tok:
            continue
        out[r["model"]].append((r["wall_s"] / tok, r["target"], tok, r["wall_s"]))
    return out


before, after = cells(recs[:CUT]), cells(recs[CUT:])
print(f"split at record {CUT}: {sum(len(v) for v in before.values())} sized ok before, "
      f"{sum(len(v) for v in after.values())} after\n")
print(f"{'generator':14}{'n_bef':>6}{'s/tok_bef':>11}{'n_aft':>6}{'s/tok_aft':>11}{'change':>9}")
signs = []
for g in sorted(set(before) | set(after)):
    b = [x[0] for x in before.get(g, [])]
    a = [x[0] for x in after.get(g, [])]
    mb = statistics.median(b) if b else float("nan")
    ma = statistics.median(a) if a else float("nan")
    ch = "--"
    if b and a and mb:
        pct = 100 * (ma - mb) / mb
        ch = f"{pct:+.0f}%"
        signs.append(pct)
    print(f"{g:14}{len(b):>6}{mb:>11.3f}{len(a):>6}{ma:>11.3f}{ch:>9}")

print("\nfolds after the split, with sizes (so a size effect is visible rather than hidden):")
for g in sorted(after):
    for spt, t, tok, w in sorted(after[g]):
        print(f"  {g:14} {t:6} tokens={tok:5} wall={w:7.1f}s  s/tok={spt:.3f}")

if signs:
    up = sum(1 for x in signs if x > 0)
    print(f"\nsign summary: {up} of {len(signs)} generator cells slower after the split "
          f"(median change {statistics.median(signs):+.0f}%)")
    print("With this few folds per cell the DIRECTION across cells is the evidence, not the "
          "magnitude of any one cell.")
else:
    print("\nno comparable cells yet")
