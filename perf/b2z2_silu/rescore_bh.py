#!/usr/bin/env python3
"""Re-read `b2z2-fusion-rebuild`'s eight committed CIFs on the axis the fixture supports.

That run scored `cdk2x2_512` whole-molecule and killed `TT_BIO_UNFUSED_SILU` at 0.805 A. The
fixture is CDK2 fused to its own first 214 residues with no inter-domain interface, so the hinge
between the two pseudo-domains saturates whole-molecule RMSD for any reassociation -- the shipped
default itself moves 8.60 A on it. Its CIFs are committed (`perf/b2z2_fusion/cif/`), so the
per-domain reading costs no card time at all: same folds, correct axis.

Blackhole, qb2 card 1, ttnn 0.68.0 -- these are that row's folds, not new ones. No seed floor here;
it has one arm per seed. The seed floor comes from this row's own WH run.

    rescore_bh.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "b2z2_fusebias"))
from score import GT, ca_map, load, native, pair                             # noqa: E402

CIF = REPO / "perf" / "b2z2_fusion" / "cif"
SPLIT = 298

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=REPO / "perf/b2z2_silu/out/rescore_bh.json")
a = ap.parse_args()

gt = ca_map(GT)
report: dict = {"source": "b2z2-fusion-rebuild, qb2 card 1 (Blackhole), ttnn 0.68.0",
                "split_seq_id": SPLIT, "sizes": {}}
for size in ("512", "298"):
    n = int(size)
    split = SPLIT if n > SPLIT else n
    dirs = sorted(d for d in CIF.iterdir() if d.name.startswith(size + "_"))
    S = {d.name[len(size) + 1:]: load(next(d.glob("*.cif")), split) for d in dirs}
    sec = {"arms": sorted(S), "n_atoms": len(next(iter(S.values()))["keys"])}
    sec["aa_floor"] = pair(S["base_1"], S["base_0"], split)
    sec["lever"] = {arm: pair(S[arm], S["base_0"], split)
                    for arm in S if arm not in ("base_0", "base_1")}
    sec["native"] = {arm: native(S[arm]["cif"], n, gt) for arm in sorted(S)}
    report["sizes"][size] = sec
    print(f"\n=== {size} aa (BH) ===")
    for arm, v in list(sec["lever"].items()) + [("A/A base_1", sec["aa_floor"])]:
        print(f"  {arm:12s} whole {v['whole_all_atom_A']:8.4f} A | d1 {v['domain1_all_atom_A']:7.4f} "
              f"| d2 {v.get('domain2_all_atom_A', 0):7.4f} | hinge {v['hinge_deg']:6.2f} deg "
              f"| lddt {v['lddt_ca']:.5f}")
    for arm, v in sec["native"].items():
        print("  native %-10s %s" % (arm, {k: (w["ca_rmsd_A"], w["lddt_ca"]) for k, w in v.items()}))

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(report, indent=1))
print("\nwrote", a.out)
