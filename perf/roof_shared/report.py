#!/usr/bin/env python3
"""Pull the numbers this task has to state out of score.py's json, so none of them is retyped.

    report.py --shared out/score_shared.json --plain out/score_plain.json --out out/report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ["domain1_all_atom_A", "domain2_all_atom_A", "hinge_free_all_atom_A",
        "domain1_ca_A", "domain2_ca_A", "whole_all_atom_A", "lddt_ca",
        "lddt_ca_domain1", "lddt_ca_domain2", "hinge_deg"]


def worst_domain(p: dict) -> float:
    """The anchor's headline reading: the worse of the two pseudo-domains, all-atom."""
    return max(v for k, v in p.items() if k.endswith("_all_atom_A") and k.startswith("domain"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=Path, required=True)
    ap.add_argument("--plain", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    S = json.loads(a.shared.read_text())
    P = json.loads(a.plain.read_text())
    rep = {"sources": {"shared": str(a.shared), "plain": str(a.plain)}, "sizes": {}}

    for size in sorted(S["sizes"], key=int):
        s, p = S["sizes"][size], P["sizes"][size]
        sec = {"n_atoms": s["n_atoms"], "fold_s": s["fold_s"], "plddt": s["plddt"]}
        sec["pairs"] = {}
        for src, label in ((s, "vs gpurefshared"), (p, "vs gpuref")):
            for tag, v in src["lever"].items():
                sec["pairs"][f"{tag} {label}"] = {
                    "worst_domain_all_atom_A": round(worst_domain(v), 5),
                    **{k: v[k] for k in KEYS if k in v}}
        sec["seed_floor"] = {k: {"worst_domain_all_atom_A": round(worst_domain(v), 5),
                                 **{kk: v[kk] for kk in KEYS if kk in v}}
                             for k, v in {**s["seed_floor"], **p["seed_floor"]}.items()}
        sec["native"] = {t: s["native"][t] for t in sorted(s["native"])}
        rep["sizes"][size] = sec

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))

    for size, sec in rep["sizes"].items():
        print(f"\n=== {size} aa ({sec['n_atoms']} atoms) ===")
        for k, v in sec["pairs"].items():
            print(f"  {k:34s} worst-domain {v['worst_domain_all_atom_A']:8.5f} A  "
                  f"lDDT(arm vs ref) {v.get('lddt_ca')}")
        print("  -- seed floor --")
        for k, v in sec["seed_floor"].items():
            print(f"  {k:34s} worst-domain {v['worst_domain_all_atom_A']:8.5f} A")
        print("  -- against 1HCL --")
        for t, doms in sec["native"].items():
            cells = "  ".join(f"{d}: lDDT {x['lddt_ca']:.5f} CA {x['ca_rmsd_A']:.5f} A"
                              for d, x in doms.items())
            print(f"  {t:20s} {cells}")
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
