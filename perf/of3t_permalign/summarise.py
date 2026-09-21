#!/usr/bin/env python3
"""Render the D117 arm table from `runs/*.json`, so no figure in the state doc is retyped."""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "runs")

HEAD = ("| arm | batch | ref spaces | with alternatives | how it was called | alignment | "
        "GT atoms moved | differs from naive | error |")
SEP = "|---|---|---|---|---|---|---|---|---|"


def describe(r: dict) -> str:
    c = r["call"]
    if c.get("via_forward"):
        s = "real forward, checkpointed"
        if len(r.get("verdict_per_pass", [])) > 1:
            s += f", {len(r['verdict_per_pass'])} passes over ONE dict"
        return s
    bits = [f"samples {c['samples']}", f"collate {c['collate']}"]
    if c.get("drop_key"):
        bits.append("key deleted")
    if c.get("flip_symmetric"):
        bits.append("prediction flipped")
    return ", ".join(bits)


def main() -> int:
    rows = []
    for p in sorted(RUNS.glob("*.json")):
        r = json.loads(p.read_text())
        v, rs, pv = r["verdict"], r["ref_spaces"], r["provenance"]
        name = pv.get("pdb_id", "5nw3 (frozen)")
        if pv.get("source") == "WeightedPDBDataset":
            name = f"{pv['pdb_id']} {pv['datapoint']} crop {pv['crop']}"
        err = v["errors"][0]["error"] if v["errors"] else ""
        n = r.get("vs_naive") or {}
        vs = ("--" if "atoms_differing_from_naive" not in n
              else f"{n['atoms_differing_from_naive']} atoms, max "
                   f"{n['max_abs_coord_diff_A']:.3f} A")
        rows.append("| `{}` | {} | {} | {} | {} | **{}** | {} | {} | {} |".format(
            r["tag"], name,
            rs.get("n_ref_spaces_sample0") or rs.get("n_entries", "?"),
            rs.get("n_ref_spaces_with_alternatives", "-"),
            describe(r),
            "COMPLETED" if v["completed"] else "FELL BACK",
            r["effect"]["gt_atoms_moved"], vs,
            f"`{err}`" if err else "--"))
    print(HEAD)
    print(SEP)
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
