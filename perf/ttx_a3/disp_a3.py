#!/usr/bin/env python3
"""All-atom superposed displacement between the 1536 aa folds this row kept, in Angstrom.

`coords` and `rmsd` are imported from `perf/roof_msa_ladder/rmsd.py` rather than restated; that
script's `main` is keyed to its own filenames, these are keyed to `perf/ttx_a3/f1536/`.

Three comparisons, and only the third is the lever:

  A/A   off vs off2, same arm, same seed, two processes. The floor any arm number has to clear.
  seed  off vs offseed, same arm, seed 0 against seed 7. The variation already accepted.
  A/B   off vs on, same seed. The above-cap fused route's own displacement.

Each is read twice. The global reading superposes the whole complex at once, so a pair of chains
that folded the same way and came to rest in a different relative position reads as a large
displacement. The per-chain reading superposes each chain on its own counterpart and takes the
worst, which is the local question: did the fold itself change. The seed floor is what makes the
difference matter -- two seeds rearrange the complex -- so both are reported rather than one.

    python3 perf/ttx_a3/disp_a3.py --dir perf/ttx_a3/f1536 --out perf/ttx_a3/disp_1536.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_msa_ladder_rmsd", REPO / "perf" / "roof_msa_ladder" / "rmsd.py")
_M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    xyz, names = {}, {}
    for f in sorted(a.dir.glob("*.cif")):
        tag = f.name.rsplit(".", 1)[0].split("_")[-1]      # off / off2 / offseed / on
        xyz[tag], names[tag] = _M.coords(f)
    n = {k: len(v) for k, v in names.items()}
    assert len(set(n.values())) == 1, f"atom counts differ: {n}"

    ref_tag = "off"
    rows = defaultdict(list)
    for i, (ch, _resi, _at) in enumerate(names[ref_tag]):
        rows[ch].append(i)

    pairs = {"AA_same_arm_two_processes": ("off", "off2"),
             "seed_floor_off_seed0_vs_seed7": ("off", "offseed"),
             "AB_route_off_vs_on_same_seed": ("off", "on")}
    out = {}
    for label, (p, q) in pairs.items():
        if p not in xyz or q not in xyz:
            continue
        per_chain = {ch: round(_M.rmsd(xyz[p][idx], xyz[q][idx]), 5)
                     for ch, idx in rows.items()}
        out[label] = {"global_all_atom": round(_M.rmsd(xyz[p], xyz[q]), 5),
                      "per_chain": per_chain,
                      "worst_chain": round(max(per_chain.values()), 5)}

    res = {"dir": str(a.dir), "atoms": next(iter(n.values())),
           "chains": {ch: len(idx) for ch, idx in rows.items()}, "readings": out}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


sys.exit(main())
