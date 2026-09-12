#!/usr/bin/env python3
"""Rebuild every figure this row publishes, from committed JSON only.

Written because two of this row's headline numbers existed only in prose. `b2z2_slabperf.json` sat
at its pre-fix values while the text quoted the post-fix ones, and the 18.027 s fold projection was
reconstructed by hand against a single-chip base when the saving it applies is mesh-side. This
campaign has had four denominator slips and every one was caught by re-deriving from primaries, so:
no number goes out of this row that this script cannot print.

    python3 perf/b2z2_dualchip/closing.py
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PUBLISHED_CELL = 20.079   # site/data/perf-512aa.json, committed fc7fed56


def load(name):
    p = HERE / f"{name}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def med(d):
    return d["summary"]["median_s"] if d else None


def trunk(d):
    return d["summary"].get("trunk_s_median") if d else None


def cif(d):
    return d["summary"]["cif_sha256_16"][0] if d else None


def flags(d):
    return (d.get("benchlocked"), d.get("diffusion_trace")) if d else (None, None)


arms = {k: load(k) for k in (
    "b2z2_fold_single", "b2z2_fold_stages", "b2z2_fold_mesh", "b2z2_fold_mesh_notrace",
    "b2z2_fold_meshtrace", "b2z2_fold_meshctl", "b2z2_fold_sharded")}
slab = load("b2z2_slabperf")

print("=" * 78)
print("FOLD ARMS (median s, n=5, cell protocol)          bench trace   trunk_s   CIF")
for k, d in arms.items():
    if not d:
        print(f"  {k:34s} MISSING")
        continue
    b, t = flags(d)
    tr = trunk(d)
    print(f"  {k:34s} {med(d):8.4f}  {str(b)[:5]:5s} {str(t)[:5]:5s} "
          f"{(f'{tr:8.4f}' if tr else '       -')}  {cif(d)}")

digests = {cif(d) for d in arms.values() if d}
print(f"\nPARITY: {len(digests)} distinct CIF across all arms -> "
      f"{'BIT-IDENTICAL' if len(digests) == 1 else 'DIVERGENT: ' + str(digests)}")

print("\n" + "=" * 78)
print("SLAB SPEEDUPS (one chip, warm, median of 7)")
if slab:
    for k, v in slab["ops"].items():
        print(f"  {k:16s} whole {v['whole_ms']:7.3f} ms  slab {v['slab_ms']:7.3f} ms  "
              f"{v['slab_speedup']:.3f}x")
    pt = slab["pair_track_total"]
    print(f"  {'PAIR-TRACK SUM':16s} whole {pt['whole_ms']:7.3f} ms  slab {pt['slab_ms']:7.3f} ms  "
          f"{pt['speedup']:.3f}x")

print("\n" + "=" * 78)
print("MEASURED RESULTS")
sharded, ctl = arms["b2z2_fold_sharded"], arms["b2z2_fold_meshctl"]
single = arms["b2z2_fold_stages"] or arms["b2z2_fold_single"]
if sharded and ctl:
    print(f"  sharded vs its matched mesh control   fold {med(ctl)/med(sharded):.4f}x   "
          f"trunk {trunk(ctl)/trunk(sharded):.4f}x  ({trunk(ctl)-trunk(sharded):+.4f} s)")
if sharded and single:
    print(f"  sharded vs ONE-CHIP fold (mesh tax included, the honest quote)"
          f"        {med(single)/med(sharded):.4f}x")
    print(f"  sharded vs published cell {PUBLISHED_CELL} s"
          f"                            {PUBLISHED_CELL/med(sharded):.4f}x")

nt, tr_arm = arms["b2z2_fold_mesh_notrace"], arms["b2z2_fold_meshtrace"]
if nt and tr_arm and flags(nt)[0] and flags(tr_arm)[0]:
    print(f"  trace on mesh (both benchlocked)      {med(nt)/med(tr_arm):.4f}x")
elif arms["b2z2_fold_mesh"] and tr_arm:
    b_nt = flags(arms["b2z2_fold_mesh"])[0]
    print(f"  trace on mesh                         {med(arms['b2z2_fold_mesh'])/med(tr_arm):.4f}x"
          f"   [UNPAIRED: untraced arm benchlocked={b_nt}]")
if nt and single:
    print(f"  mesh tax, untraced                    {med(nt)/med(single):.4f}x")
if tr_arm and single:
    print(f"  mesh tax, traced                      {med(tr_arm)/med(single):.4f}x")

print("\n" + "=" * 78)
print("FULL-CHAIN PROJECTION -- anchored to the MEASURED sharded fold, not to a block model")
if sharded and ctl and slab:
    ops = slab["ops"]
    G = 1.680   # 67.1 MB all_gather, perf/b2z2_dualchip/b2z2_probe2.json burst at i=512
    one_net = ops["transition_z"]["whole_ms"] - ops["transition_z"]["slab_ms"] - G
    pt = slab["pair_track_total"]
    full_net = (pt["whole_ms"] - pt["slab_ms"]) - 3 * G
    scale = full_net / one_net
    saved = (trunk(ctl) - trunk(sharded)) * scale
    proj_trunk = trunk(ctl) - saved
    proj_fold = med(ctl) - saved
    print(f"  one op sharded nets {one_net:6.3f} ms/block (1 gather)")
    print(f"  all five ops net    {full_net:6.3f} ms/block (3 gathers)  -> {scale:.2f}x")
    print(f"  measured trunk saving {trunk(ctl)-trunk(sharded):.4f} s x {scale:.2f} = {saved:.3f} s")
    print(f"  projected trunk {proj_trunk:.3f} s, fold {proj_fold:.3f} s")
    if single:
        print(f"  -> {med(single)/proj_fold:.4f}x vs one chip, "
              f"{PUBLISHED_CELL/proj_fold:.4f}x vs the published cell   [PROJECTED]")
print("=" * 78)
