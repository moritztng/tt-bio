#!/usr/bin/env python3
"""The per-target quality table, with each target's own seed floor printed next to it.

One row per target, one column per setting, cell = CA lDDT against that target's own (200,3,
seed 0) fold. The FLOOR column is the same number for a re-fold at the same setting with a
different seed, and it is the only thing that makes a cell readable: 96 is a pass on a target
whose floor is 91 and a fail on one whose floor is 99.9.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = [(200, 3), (150, 3), (100, 3), (75, 3), (50, 3), (25, 3),
         (200, 2), (100, 2), (50, 2), (200, 1), (100, 1), (50, 1), (200, 0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", type=Path, required=True)
    ap.add_argument("--metric", default="lddt_ca",
                    choices=["lddt_ca", "rmsd_ca_A", "rmsd_allatom_A", "plddt"])
    a = ap.parse_args()
    d = json.loads(a.frontier.read_text())

    hdr = "".join(f"{s}/{r:<1}".rjust(9) for s, r in ORDER)
    print(f"{'target':>14} {'FLOOR':>7} {'kind':>16}{hdr}")
    for t in d["targets"]:
        cells = {}
        for row in t["rows"]:
            if row["seed"] != 0 or row["role"] == "floor":
                continue
            cells[(row["steps"], row["recycles"])] = row[a.metric]
        floor = (t["floor"]["lddt_ca_min"] if a.metric == "lddt_ca" else
                 t["floor"]["rmsd_ca_max_A"] if a.metric == "rmsd_ca_A" else
                 t["floor"]["rmsd_allatom_max_A"] if a.metric == "rmsd_allatom_A" else
                 t["floor"]["plddt_spread"])
        line = ""
        for k in ORDER:
            v = cells.get(k)
            if v is None:
                line += "        -"
                continue
            if a.metric == "lddt_ca":
                mark = " " if v >= floor else "!"
            elif a.metric == "plddt":
                mark = " "
            else:
                mark = " " if v <= floor else "!"
            line += f"{v:>8.2f}{mark}" if a.metric != "plddt" else f"{v:>8.4f}{mark}"
        f = f"{floor:>7.2f}" if a.metric != "plddt" else f"{floor:>7.4f}"
        print(f"{t['target']:>14} {f} {str(t['kind']):>16}{line}")
    print("\n! = outside this target's own seed floor, i.e. worse than re-running production "
          "with a different seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
