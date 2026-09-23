#!/usr/bin/env python3
"""Score a structure against the MGX upstream reference, with the reference's own seed floor.

    python3 perf/mgx/ref/score.py --model boltz2 --fixture 3abq_1536 path/to/tt.cif
    python3 perf/mgx/ref/score.py --floors            # every cell's seed floor, as a table

For one structure it prints CA-RMSD and CA-lDDT against each reference seed, and beside them the
floor: the same two numbers between reference seed 0 and seed 1. A deviation is read against that
floor, never on its own (`a-ratio-is-not-a-defect-until-the-references-own-ratio-is-measured`).
Chains are matched by sequence, trying every assignment of identical copies, so chain letters do
not have to agree (tests/ca_rmsd.py --by-sequence).

References live in perf/mgx/ref/refs/<model>/<fixture>/s<seed>.cif.gz, described by
perf/mgx/ref/manifest.json. A cell the upstream could not fold carries its error there, and the
scorer prints it instead of a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import gemmi  # noqa: E402
from ca_rmsd import best_rmsd, ca_chains  # noqa: E402

MANIFEST = HERE / "manifest.json"
FIXTURES = HERE / "fixtures" / "fixtures.json"


def lddt_ca(model: list, ref: list, cutoff: float = 15.0) -> float:
    """Global CA-lDDT of `model` against `ref` (same order), thresholds 0.5/1/2/4 A."""
    m = np.array([[p.x, p.y, p.z] for p in model])
    r = np.array([[p.x, p.y, p.z] for p in ref])
    dr = np.linalg.norm(r[:, None] - r[None], axis=-1)
    dm = np.linalg.norm(m[:, None] - m[None], axis=-1)
    mask = (dr < cutoff) & ~np.eye(len(r), dtype=bool)
    err = np.abs(dm - dr)[mask]
    return float(np.mean([(err < t).mean() for t in (0.5, 1.0, 2.0, 4.0)]))


def resolved_only(chains: dict, crystal: dict) -> dict:
    """`chains` cut to the residues every same-sequence crystal chain resolves."""
    out = {}
    for name, (seq, pos) in chains.items():
        keep = [set(p) for s, p in crystal.values() if s == seq]
        keys = set.intersection(*keep) if keep else set(pos)
        out[name] = (seq, {k: v for k, v in pos.items() if k in keys})
    return out


def compare(pred: str, ref: str, crystal: str | None = None) -> dict:
    """CA-RMSD after superposition and CA-lDDT of `pred` against `ref`.

    With `crystal`, both sides are cut to the residues the crystal resolves first. Disordered
    termini and loops have no defined position, so upstream seeds scatter there and dominate a
    whole-chain number on a real complex."""
    a, b = ca_chains(ref), ca_chains(pred)
    if crystal:
        gt = ca_chains(crystal)
        a, b = resolved_only(a, gt), resolved_only(b, gt)
    best = best_rmsd(a, b)
    if best is None:
        return {"error": "no chain of the prediction matches the reference by sequence"}
    rmsd, n, mapping, pr, pp = best
    # Each chain superposed on its own: the fold of every copy, independent of where it sits.
    # On a tiled fixture the copies share no real interface, so only this number is meaningful.
    per_chain = []
    for na, nb in mapping:
        ks = sorted(set(a[na][1]) & set(b[nb][1]))
        if len(ks) >= 3:
            per_chain.append(gemmi.superpose_positions([a[na][1][k] for k in ks],
                                                       [b[nb][1][k] for k in ks]).rmsd)
    return {"ca_rmsd_A": round(rmsd, 3), "lddt_ca": round(lddt_ca(pp, pr), 4), "n_ca": n,
            "worst_chain_ca_rmsd_A": round(max(per_chain), 3) if per_chain else None}


def cell(manifest: dict, model: str, fixture: str) -> dict:
    return manifest.get("cells", {}).get(f"{model}/{fixture}", {})


def ref_paths(c: dict) -> dict[int, Path]:
    return {int(s): ROOT / r["cif"] for s, r in c.get("seeds", {}).items()
            if r.get("status") == "ok" and (ROOT / r["cif"]).is_file()}


def floor(c: dict, crystal: str | None = None) -> dict | None:
    """Seed 0 vs seed 1 of the reference, the number every deviation is quoted beside."""
    refs = ref_paths(c)
    if len(refs) < 2:
        return None
    s = sorted(refs)
    return compare(str(refs[s[1]]), str(refs[s[0]]), crystal)


def crystal_of(fixture: str) -> str | None:
    gt = json.loads(FIXTURES.read_text())["fixtures"].get(fixture, {}).get("ground_truth")
    return str(ROOT / gt) if gt else None


def score(model: str, fixture: str, pred: str) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    c = cell(manifest, model, fixture)
    out = {"model": model, "fixture": fixture, "pred": pred}
    refs = ref_paths(c)
    if not refs:
        out["error"] = ("no reference for this cell: " +
                        (c.get("missing_reason") or "cell not in manifest"))
        return out
    out["vs_ref"] = {f"s{s}": compare(pred, str(p)) for s, p in sorted(refs.items())}
    out["floor"] = floor(c)
    gt = crystal_of(fixture)
    if gt:
        out["resolved"] = {"vs_ref": {f"s{s}": compare(pred, str(p), gt) for s, p in sorted(refs.items())},
                           "floor": floor(c, gt)}
    rm = [v["ca_rmsd_A"] for v in out["vs_ref"].values() if "ca_rmsd_A" in v]
    if rm:
        out["mean_ca_rmsd_A"] = round(sum(rm) / len(rm), 3)
    if out["floor"] and rm and out["floor"].get("ca_rmsd_A"):
        out["ratio_to_floor"] = round(out["mean_ca_rmsd_A"] / out["floor"]["ca_rmsd_A"], 2)
    return out


def line(r: dict) -> str:
    if "error" in r:
        return f"{r['model']} {r['fixture']}: {r['error']}"
    vs = ", ".join(f"{k} {v.get('ca_rmsd_A')} A / lDDT {v.get('lddt_ca')} / chain {v.get('worst_chain_ca_rmsd_A')} A"
                   for k, v in r["vs_ref"].items())
    f = r["floor"]
    fl = (f"floor {f['ca_rmsd_A']} A / lDDT {f['lddt_ca']} / chain {f.get('worst_chain_ca_rmsd_A')} A"
          if f else "floor: single reference seed")
    ratio = f", {r['ratio_to_floor']}x floor" if "ratio_to_floor" in r else ""
    n = next(iter(r["vs_ref"].values())).get("n_ca")
    out = f"{r['model']} {r['fixture']}: vs ref {vs} | {fl}{ratio} | n_ca {n}"
    if "resolved" in r:
        rv = r["resolved"]
        vs = ", ".join(f"{k} {v.get('ca_rmsd_A')} A / lDDT {v.get('lddt_ca')}" for k, v in rv["vs_ref"].items())
        f = rv["floor"]
        fl = f"floor {f['ca_rmsd_A']} A / lDDT {f['lddt_ca']}" if f else "floor: single reference seed"
        n = next(iter(rv["vs_ref"].values())).get("n_ca")
        out += f"\n  crystal-resolved residues only: vs ref {vs} | {fl} | n_ca {n}"
    return out


def floors() -> None:
    manifest = json.loads(MANIFEST.read_text())
    fx = json.loads(FIXTURES.read_text())["fixtures"]
    models = manifest.get("models", [])
    print("| model | " + " | ".join(fx) + " |")
    print("|---|" + "---|" * len(fx))
    for m in models:
        row = []
        for f in fx:
            c = cell(manifest, m, f)
            fl = floor(c)
            if fl and "ca_rmsd_A" in fl:
                row.append(f"{fl['ca_rmsd_A']} A ({fl['lddt_ca']}, chain {fl['worst_chain_ca_rmsd_A']} A)")
            else:
                row.append(c.get("missing_reason", "no reference") if not ref_paths(c) else "1 seed")
        print(f"| {m} | " + " | ".join(row) + " |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", nargs="?", help="the structure to score (cif, cif.gz or pdb)")
    ap.add_argument("--model")
    ap.add_argument("--fixture")
    ap.add_argument("--json", action="store_true", help="print the full record as json")
    ap.add_argument("--floors", action="store_true", help="print every cell's seed floor")
    args = ap.parse_args()
    if args.floors:
        floors()
        return 0
    if not (args.pred and args.model and args.fixture):
        ap.error("pred, --model and --fixture are required unless --floors")
    r = score(args.model, args.fixture, args.pred)
    print(json.dumps(r, indent=1) if args.json else line(r))
    return 1 if "error" in r else 0


if __name__ == "__main__":
    raise SystemExit(main())
