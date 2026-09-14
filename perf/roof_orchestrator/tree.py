#!/usr/bin/env python3
"""Turn roof-budget's flat 26-row table into the call tree it actually describes.

The table mixes top-level units, intermediate units and leaves in one list. Three rows
(`DiffusionModule`, `Diffusion`, `DiffusionTransformer`) carry near-identical GFLOP/call,
which is what nesting looks like, so ANY ranking that sums rows double-counts unless the
tree is known. roof-budget's own summary sums exactly three rows and no more.

Parent/child is asserted here from call-count ratios and time containment, then CHECKED:
a parent's children must sum to <= its own time, and the residual is reported rather than
hidden. No new measurement -- this reads the committed table only.
"""
import json
import sys
from pathlib import Path

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rb.json"
d = json.load(open(SRC))
rows = {r["sig"]: r for r in d["rows"]}
S = d["summary"]

# (child, parent, share) -- share splits a row whose calls serve two parents.
# Transition|1x512x512x128 is 280 calls = the trunk pairformer's 264 + the MSA-side
# pairformer's 16, the only row in the table that spans two parents.
EDGES = [
    ("PairformerLayer|1x512x384,1x512x512x128", None, 1.0),
    ("TriangleMultiplication|1x512x512x128,1x512x512", "PairformerLayer|1x512x384,1x512x512x128", 1.0),
    ("TriangleAttention|1x512x512x128,1x1x1x512",      "PairformerLayer|1x512x384,1x512x512x128", 1.0),
    ("Transition|1x512x512x128",                       "PairformerLayer|1x512x384,1x512x512x128", 264/280),
    ("AttentionPairBias|1x512x384,1x512x512x128",      "PairformerLayer|1x512x384,1x512x512x128", 1.0),
    ("Transition|1x512x384",                           "PairformerLayer|1x512x384,1x512x512x128", 1.0),

    ("DiffusionModule|", None, 1.0),
    ("Diffusion|1x4480x3,1", "DiffusionModule|", 1.0),
    ("DiffusionTransformer|1x512x768,1x512x768",  "Diffusion|1x4480x3,1", 1.0),
    ("DiffusionTransformer|1x140x32x128,1x140x32x128", "Diffusion|1x4480x3,1", 1.0),
    ("Transition|1x512x768", "Diffusion|1x4480x3,1", 1.0),
    ("DiffusionTransformerLayer|1x512x768,1x512x768", "DiffusionTransformer|1x512x768,1x512x768", 1.0),
    ("ConditionedTransitionBlock|1x512x768,1x512x768", "DiffusionTransformerLayer|1x512x768,1x512x768", 1.0),
    ("AttentionPairBias|1x512x768,1x16x512x512",       "DiffusionTransformerLayer|1x512x768,1x512x768", 1.0),
    ("AdaLN|1x512x768,1x512x768",                      "DiffusionTransformerLayer|1x512x768,1x512x768", 1.0),
    ("DiffusionTransformerLayer|1x140x32x128,1x140x32x128", "DiffusionTransformer|1x140x32x128,1x140x32x128", 1.0),
    ("AttentionPairBias|1x140x32x128,140x4x32x128",         "DiffusionTransformerLayer|1x140x32x128,1x140x32x128", 1.0),
    ("ConditionedTransitionBlock|1x140x32x128,1x140x32x128","DiffusionTransformerLayer|1x140x32x128,1x140x32x128", 1.0),
    ("AdaLN|1x140x32x128,1x140x32x128",                     "DiffusionTransformerLayer|1x140x32x128,1x140x32x128", 1.0),

    ("MSALayer|1x512x512x128,1x1024x512x64", None, 1.0),
    ("OuterProductMean|1x1024x512x64,1024x1x1",         "MSALayer|1x512x512x128,1x1024x512x64", 1.0),
    ("PairformerLayer|1x512x512x128",                   "MSALayer|1x512x512x128,1x1024x512x64", 1.0),
    ("PairWeightedAveraging|1x1024x512x64,1x512x512x128","MSALayer|1x512x512x128,1x1024x512x64", 1.0),
    ("Transition|1x1024x512x64",                        "MSALayer|1x512x512x128,1x1024x512x64", 1.0),
    ("TriangleMultiplication|1x512x512x128", "PairformerLayer|1x512x512x128", 1.0),
    ("TriangleAttention|1x512x512x128",      "PairformerLayer|1x512x512x128", 1.0),
    ("Transition|1x512x512x128",             "PairformerLayer|1x512x512x128", 16/280),
]

assert {c for c, _, _ in EDGES} | {p for _, p, _ in EDGES if p} <= set(rows), "sig not in table"
kids = {}
for c, p, sh in EDGES:
    kids.setdefault(p, []).append((c, sh))
parents = {p for _, p, _ in EDGES if p}
tops = [c for c, p, _ in EDGES if p is None]


def t(sig, share=1.0):
    return rows[sig]["s_per_fold"] * share


def above(sig, share=1.0):
    return rows[sig]["s_above_roof_at_cell"] * share


def walk(sig, share, depth, out):
    r = rows[sig]
    ch = kids.get(sig, [])
    csum = sum(t(c, s) for c, s in ch)
    # residual in the SAME units as the above-roof column: s/fold scaled to the cell by the
    # table's own single factor. Reporting it in session seconds next to cell seconds would be
    # the composition error this campaign exists to catch.
    resid = (t(sig, share) - csum * share) * S["cell_scale"]
    out.append((depth, sig, r["calls"], t(sig, share), above(sig, share),
                resid if ch else None, r["pct_binding_roof"], r["binding_roof"]))
    for c, s in sorted(ch, key=lambda x: -t(*x)):
        walk(c, s * share, depth + 1, out)


out = []
for s in sorted(tops, key=lambda x: -t(x)):
    walk(s, 1.0, 0, out)

print(f"# the fold's call tree, from {S['head']} -- session fold {S['session_fold_s']} s, "
      f"cell {S['cell_of_record_s']} s, floor {S['binding_floor_s']} s ({S['binding_roof']})")
print(f"# roofs measured: {S['stream_roof_GBps']} GB/s stream, {S['compute_roof_TFLOPs']} TFLOP/s "
      f"dense bf16, balance {S['machine_balance_flop_per_byte']} FLOP/byte\n")
print("# unattr* = parent time minus the sum of its measured children, scaled to the cell\n")
print(f"{'unit':<58} {'calls':>6} {'s/fold':>7} {'s>roof':>7} {'unattr*':>7} {'%roof':>6} roof")
for depth, sig, calls, sf, ab, resid, pct, roof in out:
    ind = "  " * depth + ("" if depth == 0 else "- ")
    rs = f"{resid:7.3f}" if resid is not None else "   leaf"
    print(f"{ind + sig:<58.58} {calls:>6} {sf:7.3f} {ab:7.3f} {rs} {pct:6.1f} {roof}")

leaves = [(sig, sh) for sig, sh in
          {(c, s) for c, s in [(c, s) for c, p, s in
           [(c, p, s) for c, p, s in EDGES]] if c not in parents}]
seen = {}
for c, p, s in EDGES:
    if c not in parents:
        seen[c] = seen.get(c, 0.0) + s
print(f"\n# LEAVES ONLY -- the only rows that can be ranked without double-counting")
tot = 0.0
for sig, sh in sorted(seen.items(), key=lambda kv: -above(*kv)):
    tot += above(sig, sh)
    print(f"  {sig:<56.56} {above(sig, sh):6.3f} s above roof   {rows[sig]['pct_binding_roof']:5.1f} % of {rows[sig]['binding_roof']} roof")
print(f"  {'leaf total':<56} {tot:6.3f} s")
unattr = sum(r for _, _, _, _, _, r, _, _ in out if r is not None)
print(f"  {'unattributed inside parents, at the cell':<56} {unattr:6.3f} s")
print(f"  {'leaves + unattributed':<56} {tot + unattr:6.3f} s")
print(f"  {'cell above floor':<56} {S['cell_s_above_roof']:6.3f} s")
print(f"  {'closure gap (the at-roof share of the unattributed)':<56} "
      f"{tot + unattr - S['cell_s_above_roof']:6.3f} s "
      f"= {100*abs(tot+unattr-S['cell_s_above_roof'])/S['cell_s_above_roof']:.1f} % of the prize")
