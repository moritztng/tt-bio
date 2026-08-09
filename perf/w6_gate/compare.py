#!/usr/bin/env python3
"""Score every W6 gate arm against BASE at the same seed. Card-free.

The acceptance band is BASE-vs-CTRL, measured, not assumed. CTRL is the same fold with the
256<seq<=384 special case deleted, so N=320 falls back to the 256/256 config this file already
emits at every other size. BASE-vs-CTRL is therefore the distance two *shipped* reduction orders
put between two folds of the same target at this size, with real inputs. An arm inside that band
is inside the noise the code already produces; an arm outside it is a real change.

Kabsch and TM come from tests/test_structure.py, the same helpers release_gate.py uses.

    python3 perf/w6_gate/compare.py [--md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "out"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from test_structure import _kabsch_deviations, _tm_score, get_ca_atoms  # noqa: E402

ARM_ORDER = ["BASE", "CTRL", "C1", "C2", "C2FIX", "C3", "C4", "ALL", "ALLFIX"]


def _ca(tag: str) -> np.ndarray | None:
    """CA coordinates of the rank-0 model, in entity-position order."""
    cifs = sorted((OUT / tag).glob("*.cif"))
    best = [c for c in cifs if "_model_" not in c.name] or cifs
    if not best:
        return None
    chains = get_ca_atoms(str(best[0]))
    rows = []
    for cid in sorted(chains):
        rows += [chains[cid][p] for p in sorted(chains[cid])]
    return np.asarray(rows, dtype=float)


def _gt_ca() -> np.ndarray | None:
    p = HERE / "gt" / "1hcl.cif"
    if not p.is_file():
        return None
    chains = get_ca_atoms(str(p))
    cid = max(chains, key=lambda c: len(chains[c]))
    return np.asarray([chains[cid][k] for k in sorted(chains[cid])], dtype=float)


def _vs_gt(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float] | None:
    """CA-RMSD / TM against the ground truth, on the overlapping prefix.

    1HCL resolves fewer residues than CDK2 has, so an index-wise pairing is only honest when the
    two have the same length. Say N/A rather than align approximately and quote a number.
    """
    if gt is None or pred is None or len(gt) != len(pred):
        return None
    dev = _kabsch_deviations(pred, gt)
    return float(np.sqrt((dev ** 2).mean())), _tm_score(dev, len(gt))


def score(model: str, size: str) -> list[dict] | None:
    recs = {}
    for arm in ARM_ORDER:
        p = OUT / f"{arm}_{model}_{size}.json"
        if p.is_file():
            recs[arm] = json.loads(p.read_text())
    if "BASE" not in recs:
        return None

    base_xyz = np.load(OUT / f"BASE_{model}_{size}" / "coords.npy")
    base_ca = _ca(f"BASE_{model}_{size}")
    base = recs["BASE"]
    gt = _gt_ca() if size == "298" else None

    rows = []
    for arm, r in recs.items():
        d = OUT / f"{arm}_{model}_{size}"
        xyz = np.load(d / "coords.npy") if (d / "coords.npy").is_file() else None
        ca = _ca(f"{arm}_{model}_{size}")
        row = {"arm": arm}
        if xyz is not None and xyz.shape == base_xyz.shape:
            row["exact"] = bool(np.array_equal(xyz, base_xyz))
            row["max_abs_delta_A"] = float(np.abs(xyz - base_xyz).max())
        if ca is not None and base_ca is not None and ca.shape == base_ca.shape:
            dev = _kabsch_deviations(ca, base_ca)
            row["ca_rmsd_vs_base_A"] = float(np.sqrt((dev ** 2).mean()))
            row["tm_vs_base"] = _tm_score(dev, len(base_ca))
        row["plddt"] = r["confidence"].get("plddt")
        row["d_plddt"] = round(r["confidence"].get("plddt", 0) - base["confidence"]["plddt"], 6)
        row["ptm"] = r["confidence"].get("ptm")
        row["d_ptm"] = round(r["confidence"].get("ptm", 0) - base["confidence"]["ptm"], 6)
        row["warm_median_s"] = r["warm_median_s"]
        row["speedup_vs_base"] = (round(base["warm_median_s"] / r["warm_median_s"], 4)
                                  if r["warm_median_s"] else None)
        g = _vs_gt(ca, gt)
        if g:
            row["ca_rmsd_vs_1hcl_A"], row["tm_vs_1hcl"] = round(g[0], 3), round(g[1], 4)
        rows.append(row)

    band = next((r for r in rows if r["arm"] == "CTRL"), None)
    for r in rows:
        if r["arm"] == "BASE":
            r["verdict"] = "reference"
        elif r.get("exact"):
            r["verdict"] = "bit-exact"
        elif band is None:
            r["verdict"] = "no control band measured"
        elif r["arm"] == "CTRL":
            r["verdict"] = "control band"
        else:
            inside = (r.get("ca_rmsd_vs_base_A", 9e9) <= band.get("ca_rmsd_vs_base_A", 0)
                      and abs(r["d_plddt"]) <= abs(band["d_plddt"]))
            r["verdict"] = "inside envelope" if inside else "outside control band"
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="markdown tables for the state doc")
    args = ap.parse_args()

    allrows = {}
    for model in ("protenix-v2", "opendde"):
        for size in ("298", "117"):
            rows = score(model, size)
            if not rows:
                continue
            allrows[f"{model}/{size}"] = rows
            if args.md:
                print(f"\n**{model}, {size} aa** (BASE warm median "
                      f"{rows[0]['warm_median_s']} s)\n")
                cols = ["arm", "exact", "max_abs_delta_A", "ca_rmsd_vs_base_A", "tm_vs_base",
                        "d_plddt", "d_ptm", "warm_median_s", "speedup_vs_base",
                        "ca_rmsd_vs_1hcl_A", "tm_vs_1hcl", "verdict"]
                cols = [c for c in cols if any(c in r for r in rows)]
                print("| " + " | ".join(cols) + " |")
                print("|" + "---|" * len(cols))
                for r in rows:
                    def f(c):
                        v = r.get(c)
                        if v is None:
                            return "-"
                        if isinstance(v, float):
                            return f"{v:.4g}"
                        return str(v)
                    print("| " + " | ".join(f(c) for c in cols) + " |")
            else:
                print(f"== {model} {size} aa ==")
                for r in rows:
                    print(json.dumps(r))
    (OUT / "compare.json").write_text(json.dumps(allrows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
