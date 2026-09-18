#!/usr/bin/env python3
"""Accuracy cost of TT_BIO_PAIR_B8: paired base-vs-on RMSD with the run's own A/A floor beside it.

Parser and superposition come from `perf/other512/cif_rmsd.py` (all-atom Kabsch, equal weights) so
the Angstrom numbers are comparable with the rest of this lineage. That scorer's `arm_of` regex
reads `<size>_<arm>_<run>`; this row's harness writes `<size>_<arm>_<block>_<fold>`, so the arm is
taken from field 2 here and the pairs are grouped properly:

  * A/B  -- base vs on, same seed, same process shape. This is the reported cost.
  * A/A  -- base vs base across blocks. Boltz-2 is deterministic per seed, so this SHOULD be
            0.000000; anything else means the instrument moved and no other number is readable.

plDDT comes from the A/B json, not from the CIFs, and is reported as the paired drop.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _lineage_scorer():
    p = REPO / "perf" / "other512" / "cif_rmsd.py"
    spec = importlib.util.spec_from_file_location("_cif_rmsd", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifdir", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--ab", default=None, help="the fold A/B json, for the plDDT pairing")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = _lineage_scorer()

    arms: dict[str, list[tuple[str, object]]] = {}
    for d in sorted(Path(a.cifdir).iterdir()):
        if not d.is_dir() or not d.name.startswith(f"{a.size}_"):
            continue
        f = d.name.split("_")
        if len(f) < 3:
            continue
        arm = f[1]
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            continue
        keys, xyz = S.read_atoms(cifs[0])
        arms.setdefault(arm, []).append((d.name, (keys, xyz)))

    res: dict = {"size": a.size, "n_per_arm": {k: len(v) for k, v in arms.items()}}
    if not arms.get("base") or not arms.get("on"):
        res["error"] = f"need both arms on disk, have {sorted(arms)}"
        print(json.dumps(res, indent=1))
        return 1

    ref_keys = arms["base"][0][1][0]
    for arm, rows in arms.items():
        for name, (k, _) in rows:
            if k != ref_keys:
                res["error"] = f"atom identity differs in {name}; cannot compare atom-for-atom"
                print(json.dumps(res, indent=1))
                return 1
    res["atoms"] = len(ref_keys)

    def pairs(xs, ys=None):
        it = itertools.combinations(xs, 2) if ys is None else itertools.product(xs, ys)
        return [(x[0], y[0], S.kabsch_rmsd(x[1][1], y[1][1])) for x, y in it]

    aa = pairs(arms["base"])
    bb = pairs(arms["on"])
    ab = pairs(arms["base"], arms["on"])
    res["aa_base_vs_base"] = {"max_A": round(max((r for _, _, r in aa), default=0.0), 6),
                              "n": len(aa)}
    res["aa_on_vs_on"] = {"max_A": round(max((r for _, _, r in bb), default=0.0), 6), "n": len(bb)}
    rs = [r for _, _, r in ab]
    res["ab_base_vs_on"] = {"median_A": round(st.median(rs), 6), "min_A": round(min(rs), 6),
                            "max_A": round(max(rs), 6), "n": len(rs)}
    res["pairs"] = [{"a": x, "b": y, "rmsd_A": round(r, 6)} for x, y, r in ab]

    if a.ab:
        d = json.loads(Path(a.ab).read_text())
        pl = {"base": [], "on": []}
        for row in d.get("blocks", []):
            if row.get("size") != a.size or not row.get("result"):
                continue
            for f in row["result"]["folds"]:
                if f.get("plddt") is not None:
                    pl[row["arm"]].append(f["plddt"])
        if pl["base"] and pl["on"]:
            mb, mo = st.median(pl["base"]), st.median(pl["on"])
            res["plddt"] = {"base_median": round(mb, 6), "on_median": round(mo, 6),
                            "drop": round(mb - mo, 6), "n_base": len(pl["base"]),
                            "n_on": len(pl["on"]),
                            "base_spread": round(max(pl["base"]) - min(pl["base"]), 6),
                            "on_spread": round(max(pl["on"]) - min(pl["on"]), 6)}

    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
