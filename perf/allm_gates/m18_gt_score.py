#!/usr/bin/env python3
"""Distance to the EXPERIMENTAL structure for each M18 arm, which is the number that decides it.

`cif_rmsd.py` answers "how far did the structure move", and 8.676893 A is a large answer. It does
not answer "did it get worse": the off arm is the incumbent route, not ground truth, so a divergence
magnitude carries no direction (`relative-l2-alone-cannot-identify-the-error-direction`). RF3 scored
the identical switch against a deposited structure and accuracy IMPROVED, 0.2030 -> 0.1780 A CA on
7ROA L117, so the direction has to be measured here rather than assumed from the size of the move.

`cdk2x2_512` is CDK2 followed by its own residues 1-214, so both halves score against 1HCL. Segments
and scorer are `perf/fused_sdpa/of3_score_ref.py`'s, unchanged, so these are comparable to what is
already on record.
"""
import sys
from pathlib import Path

ROOT = Path("/home/ttuser/.coworker/wt/allm-gates")
sys.path.insert(0, str(ROOT / "perf" / "other512"))
sys.path.insert(0, str(ROOT / "perf" / "fused_sdpa"))
from of3_score_ref import ca_map, gt_rmsd, GT_SEGMENTS  # noqa: E402

GT = ca_map(ROOT / "perf" / "fused_sdpa" / "cifs" / "1hcl.cif")
print(f"1HCL CA positions parsed: {len(GT)}\n")

rows = []
for d in sorted(list((ROOT / "perf/allm_gates/m18_cifs").glob("512_*")) +
                list((ROOT / "perf/allm_gates/m18_cifs_site").glob("512_*"))):
    cif = next(iter(d.glob("*.cif")), None)
    if cif is None:
        continue
    arm = d.name.split("_")[1]
    for label, pairs in GT_SEGMENTS[512].items():
        r, n = gt_rmsd(cif, GT, pairs)
        rows.append((d.name, arm, label, r, n))

print(f"{'run':22s} {'arm':9s} {'segment':22s} {'CA RMSD vs 1HCL':>16s} {'n':>5s}")
for name, arm, label, r, n in rows:
    print(f"{name:22s} {arm:9s} {label:22s} {r:16.6f} {n:5d}")

print("\nBY ARM, mean over both copies (lower = closer to the experimental structure):")
import collections
agg = collections.defaultdict(list)
for _, arm, _, r, _ in rows:
    agg[arm].append(r)
base = None
for arm in sorted(agg):
    m = sum(agg[arm]) / len(agg[arm])
    if arm == "off":
        base = m
for arm in sorted(agg):
    m = sum(agg[arm]) / len(agg[arm])
    d = "" if base is None or arm == "off" else f"   {m - base:+.6f} A vs off"
    print(f"  {arm:10s} {m:10.6f} A   (n={len(agg[arm])} segment-scores){d}")
