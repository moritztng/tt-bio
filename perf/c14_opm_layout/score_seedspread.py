#!/usr/bin/env python3
"""Re-score the OPM seed-spread folds with the instrument cdk2x2_512 actually supports.

Pass 1 read 11.6870 A whole-structure all-atom between the arms and called the accuracy check
failed. That is the wrong reading for this fixture and the campaign already knows it:
`perf/k10_anchor/FINDINGS.md` measures the same whole-structure column swinging 1.9 - 11.9 A on the
upstream fp32 reference against itself, because cdk2x2_512 is CDK2 followed by its own residues
1-214 with no interface between the copies, so the hinge saturates any whole-molecule RMSD for a
reassociation that moved no atom within either domain.

This reuses `perf/b2z2_fusebias/score.py`'s own functions, unmodified, so the numbers are
comparable with every other accuracy verdict in the campaign:

  domain1 / domain2 all-atom, and hinge_free_all_atom  the reading the fixture supports
  lddt_ca                                              superposition-free, the hinge cannot reach it
  vs 1HCL                                              the experimental answer, which no sampler
                                                       argument can reach

Three comparisons, each against the floor that makes it mean something:
  A(s) vs B(s)     the lever, at four seeds
  A(s) vs A(s')    the same arm at a different seed -- the floor the lever must sit under
  arm vs 1HCL      whether the fast path is further from the crystal than main is
"""
from __future__ import annotations

import itertools
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "b2z2_fusebias"))
import score as S  # noqa: E402

SPLIT = 298
CIFDIR = REPO / "perf" / "c14_opm_layout" / "cifs_seedspread"
RUNS = REPO / "perf" / "c14_opm_layout" / "seed_spread_qb2c1.json"
OUT = REPO / "perf" / "c14_opm_layout" / "seed_spread_scored.json"

KEYS = ("domain1_all_atom_A", "domain2_all_atom_A", "hinge_free_all_atom_A",
        "whole_all_atom_A", "hinge_deg", "lddt_ca", "lddt_ca_domain1", "lddt_ca_domain2")


def worst_domain(p):
    return round(max(p["domain1_all_atom_A"], p["domain2_all_atom_A"]), 5)


def main() -> int:
    runs = json.loads(RUNS.read_text())
    seeds = runs["env"]["seeds"]
    loaded = {}
    for arm in ("A", "B"):
        for s in seeds:
            cif = next((CIFDIR / f"512_{arm}_s{s}").glob("*.cif"))
            loaded[(arm, s)] = S.load(cif, SPLIT)

    gt = S.ca_map(S.GT)
    rep = {"fixture": "cdk2x2_512", "split_seq_id": SPLIT, "gt": S.GT.name,
           "seeds": seeds, "n_atoms": len(loaded[("A", seeds[0])]["keys"]),
           "arms": {"A": "TT_BIO_OPM_LEGACY_LAYOUT=1, current main",
                    "B": "TT_BIO_OPM_LEGACY_LAYOUT=0, the fast path this branch ships"}}

    lever = {}
    for s in seeds:
        p = S.pair(loaded[("B", s)], loaded[("A", s)], SPLIT)
        p["worst_domain_all_atom_A"] = worst_domain(p)
        lever[f"s{s}"] = {k: p[k] for k in KEYS if k in p} | {
            "worst_domain_all_atom_A": p["worst_domain_all_atom_A"]}

    floor = {}
    for arm in ("A", "B"):
        for x, y in itertools.combinations(seeds, 2):
            p = S.pair(loaded[(arm, y)], loaded[(arm, x)], SPLIT)
            p["worst_domain_all_atom_A"] = worst_domain(p)
            floor[f"{arm}_s{x}_vs_s{y}"] = {k: p[k] for k in KEYS if k in p} | {
                "worst_domain_all_atom_A": p["worst_domain_all_atom_A"]}

    nat = {f"{arm}_s{s}": S.native(loaded[(arm, s)]["cif"], 512, gt)
           for arm in ("A", "B") for s in seeds}

    def col(d, key):
        return [v[key] for v in d.values()]

    def stat(v):
        return {"n": len(v), "min": round(min(v), 5), "max": round(max(v), 5),
                "mean": round(st.mean(v), 5), "median": round(st.median(v), 5)}

    a_floor = {k: v for k, v in floor.items() if k.startswith("A_")}
    b_floor = {k: v for k, v in floor.items() if k.startswith("B_")}
    rep["lever_B_vs_A_per_seed"] = lever
    rep["seed_floor_same_arm"] = floor
    rep["vs_1hcl"] = nat
    rep["summary"] = {
        "lever_worst_domain_all_atom_A": stat(col(lever, "worst_domain_all_atom_A")),
        "lever_hinge_free_all_atom_A": stat(col(lever, "hinge_free_all_atom_A")),
        "lever_whole_all_atom_A": stat(col(lever, "whole_all_atom_A")),
        "lever_lddt_ca": stat(col(lever, "lddt_ca")),
        "A_seedfloor_worst_domain_all_atom_A": stat(col(a_floor, "worst_domain_all_atom_A")),
        "A_seedfloor_whole_all_atom_A": stat(col(a_floor, "whole_all_atom_A")),
        "A_seedfloor_lddt_ca": stat(col(a_floor, "lddt_ca")),
        "B_seedfloor_worst_domain_all_atom_A": stat(col(b_floor, "worst_domain_all_atom_A")),
    }
    for dom in ("copy1(res 1-298)", "copy2(res 299-512)"):
        for arm in ("A", "B"):
            rep["summary"][f"{arm}_vs_1hcl_{dom}_lddt_ca"] = stat(
                [nat[f"{arm}_s{s}"][dom]["lddt_ca"] for s in seeds])
            rep["summary"][f"{arm}_vs_1hcl_{dom}_ca_rmsd_A"] = stat(
                [nat[f"{arm}_s{s}"][dom]["ca_rmsd_A"] for s in seeds])
    rep["bar"] = {"kill_bar_A": 0.60,
                  "note": "the 0.60 A bar is a per-pseudo-domain reading on this fixture; the "
                          "whole-structure column is the free hinge and is reported for contrast "
                          "only"}
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["summary"], indent=1))
    print("\nper-seed lever (B vs A):")
    for k, v in lever.items():
        print("  %-4s worst-domain %8.5f  hinge-free %8.5f  whole %9.5f  hinge %7.2f deg  lDDT %.5f"
              % (k, v["worst_domain_all_atom_A"], v["hinge_free_all_atom_A"],
                 v["whole_all_atom_A"], v["hinge_deg"], v["lddt_ca"]))
    print("\nA's own seed floor:")
    for k, v in a_floor.items():
        print("  %-14s worst-domain %8.5f  whole %9.5f  hinge %7.2f deg  lDDT %.5f"
              % (k, v["worst_domain_all_atom_A"], v["whole_all_atom_A"],
                 v["hinge_deg"], v["lddt_ca"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
