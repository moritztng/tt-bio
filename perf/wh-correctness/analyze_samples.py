#!/usr/bin/env python3
"""`diffusion_samples: 5` must return five different structures, not one written five times.

`check_structure.py` grades each returned file on its own, so a model that ignored the knob and
wrote the same fold five times passes every per-file check. The bug is only visible across the
five, which is what this grades:

  1. sha256 per file. Any two equal -> FAIL, duplicated sample, nothing else to analyse.
  2. Pairwise CA-RMSD after Kabsch superposition, all 10 pairs, each pair superposed
     independently -- Kabsch is not transitive, so aligning everything to sample 1 and comparing
     would report distances that depend on the choice of reference
     (memory `abag-kabsch-nontransitive-and-state-gitignore-preregistration-gap`).
     FAIL if any pair is < 0.05 A.

The 0.05 A bar is below the bf16 coordinate noise floor, so it fires on "the same structure twice"
and never on two genuinely different diffusion draws. A high bar like 1.0 A would manufacture a
failure: two samples of a small rigid 64 aa domain can legitimately agree to well under 1 A.

    analyze_samples.py var_samples5_boltz2 [var_samples5_protenix-v2 ...]

Exit 0 = every cell distinct, 1 = at least one FAIL, 2 = nothing to grade.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import gemmi
import numpy as np

HERE = Path(__file__).resolve().parent
MIN_RMSD = 0.05  # A


def ca_coords(path: Path) -> np.ndarray:
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    return np.array([[a.pos.x, a.pos.y, a.pos.z]
                     for ch in st[0] for r in ch for a in r if a.name == "CA"])


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD after the optimal rigid superposition of p onto q. Both are centred first;
    the reflection guard is the standard sign fix on the last singular vector, without
    which a mirror image can score as a perfect fit."""
    p, q = p - p.mean(0), q - q.mean(0)
    v, _, wt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(v @ wt))
    r = v @ np.diag([1.0, 1.0, d]) @ wt
    return float(np.sqrt((((p @ r) - q) ** 2).sum() / len(p)))


def grade(cell_dir: Path) -> dict:
    files = sorted(f for f in cell_dir.rglob("*") if f.suffix in (".cif", ".pdb"))
    rep = {"cell": cell_dir.name, "files": [f.name for f in files], "fail": [], "warn": []}
    if len(files) < 2:
        rep["fail"].append(f"{len(files)} structure(s) returned, expected 5 samples")
        rep["verdict"] = "FAIL"
        return rep

    digests = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files}
    rep["sha256"] = digests
    for a, b in itertools.combinations(files, 2):
        if digests[a.name] == digests[b.name]:
            rep["fail"].append(f"{a.name} and {b.name} are byte-identical -- duplicated sample")

    coords = {f.name: ca_coords(f) for f in files}
    rmsd = {}
    for a, b in itertools.combinations(files, 2):
        x, y = coords[a.name], coords[b.name]
        if x.shape != y.shape or not len(x):
            rep["fail"].append(f"{a.name} has {len(x)} CA, {b.name} has {len(y)} "
                               f"-- samples of the same target must have the same length")
            continue
        v = kabsch_rmsd(x, y)
        rmsd[f"{a.name}|{b.name}"] = round(v, 4)
        if v < MIN_RMSD:
            rep["fail"].append(f"{a.name} vs {b.name}: CA-RMSD {v:.4f} A < {MIN_RMSD} A "
                               f"-- the same structure returned twice")
    rep["pairwise_ca_rmsd"] = rmsd
    if rmsd:
        rep["rmsd_min"], rep["rmsd_max"] = min(rmsd.values()), max(rmsd.values())
    if len(files) != 5:
        rep["warn"].append(f"{len(files)} structures returned, the cell asked for 5")
    rep["verdict"] = "FAIL" if rep["fail"] else ("WARN" if rep["warn"] else "PASS")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=["var_samples5_boltz2", "var_samples5_protenix-v2"])
    ap.add_argument("--artifacts", type=Path, default=HERE / "results" / "artifacts")
    ap.add_argument("--json", type=Path, default=HERE / "results" / "sample_distinctness.json")
    a = ap.parse_args()

    out, bad, seen = [], False, False
    for cell in a.cells:
        d = a.artifacts / cell
        if not d.exists():
            print(f"NO DATA: {d} -- run --group variants first")
            continue
        seen = True
        rep = grade(d)
        out.append(rep)
        print(f"{rep['verdict']}  {cell}  {len(rep['files'])} structures"
              + (f"  CA-RMSD {rep['rmsd_min']}-{rep['rmsd_max']} A" if "rmsd_min" in rep else ""))
        for f in rep["fail"]:
            print("  FAIL " + f)
        for w in rep["warn"]:
            print("  WARN " + w)
        for k, v in rep.get("pairwise_ca_rmsd", {}).items():
            print(f"    {k}  {v} A")
        bad |= rep["verdict"] == "FAIL"
    if out:
        a.json.write_text(json.dumps(out, indent=1))
    return (1 if bad else 0) if seen else 2


if __name__ == "__main__":
    sys.exit(main())
