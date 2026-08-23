#!/usr/bin/env python3
"""Score the OpenFold3 `on` and `P` arms against the GPU reference, with a scale to read the
margin against and an absolute anchor that no RNG argument can reach.

Three metrics, and the order matters:

  X   all-atom Kabsch RMSD, arm vs each reference seed. The number the task asks for. Same
      parser and same superposition as perf/other512/cif_rmsd.py, so it is comparable to the
      5.293 A / 16.321 A already on record.
  R   the reference's own spread: every reference-seed pair, same input, same code, same box.
      A margin |X_on - X_P| smaller than R is not a finding, it is the sampler's own noise.
      Same-seed different-GPU (H200 vs B200) is reported separately as the hardware floor.
  GT  CA Kabsch against the experimental structure, 1HCL. cdk2x2_298 IS CDK2 1:1 and
      cdk2x2_512 is CDK2 followed by its own first 214 residues, so both fixtures have a
      native answer. Cross-implementation same-seed RMSD carries a basin difference that no
      shared-RNG argument removes; distance-to-native does not care which basin anyone drew.

    python3 perf/of3_ref/score.py --refdir perf/of3_ref/ref --out perf/of3_ref/score.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
from cif_rmsd import kabsch_rmsd, read_atoms  # noqa: E402

CIFS = Path(__file__).resolve().parent / "cifs"
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
# Fold residue -> 1HCL residue. cdk2x2_298 is CDK2 at 1:1; cdk2x2_512 is that same chain
# followed by its own residues 1-214, so the chimera has two independently scoreable copies.
GT_SEGMENTS = {
    298: {"whole": [(i, i) for i in range(1, 299)]},
    512: {"copy1(res 1-298)": [(i, i) for i in range(1, 299)],
          "copy2(res 299-512)": [(298 + j, j) for j in range(1, 215)]},
}


def ca_map(path: Path) -> dict[int, tuple[str, np.ndarray]]:
    """{label_seq_id: (comp_id, xyz)} over CA atoms. Skips unresolved rows by construction."""
    txt = path.read_text().splitlines()
    cols = [l.strip() for l in txt if l.strip().startswith("_atom_site.")]
    idx = {c: i for i, c in enumerate(cols)}
    out: dict[int, tuple[str, np.ndarray]] = {}
    for line in txt:
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        if f[idx["_atom_site.label_atom_id"]] != "CA":
            continue
        try:
            sid = int(f[idx["_atom_site.label_seq_id"]])
        except ValueError:
            continue
        out[sid] = (f[idx["_atom_site.label_comp_id"]],
                    np.array([float(f[idx["_atom_site.Cartn_" + c]]) for c in "xyz"]))
    return out


def ca_map_chains(path: Path) -> dict[tuple[str, int], tuple[str, np.ndarray]]:
    """{(label_asym_id, label_seq_id): (comp_id, xyz)} over CA atoms of model 1.

    `ca_map` above keys on label_seq_id alone, which collides across chains -- 578 collisions in
    1AO6 (two copies of HSA) and 58 in 9BK6. It is left exactly as it is because RF3's committed
    disto_score_298.json / _512.json are scored through it on a single-chain fixture where the
    collision cannot arise. Anything multi-chain uses this instead.
    """
    txt = path.read_text().splitlines()
    cols = [l.strip() for l in txt if l.strip().startswith("_atom_site.")]
    idx = {c: i for i, c in enumerate(cols)}
    model_col = idx.get("_atom_site.pdbx_PDB_model_num")
    out: dict[tuple[str, int], tuple[str, np.ndarray]] = {}
    for line in txt:
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        if f[idx["_atom_site.label_atom_id"]] != "CA":
            continue
        if model_col is not None and f[model_col] != "1":
            continue
        try:
            sid = int(f[idx["_atom_site.label_seq_id"]])
        except ValueError:
            continue
        key = (f[idx["_atom_site.label_asym_id"]], sid)
        if key in out:      # altloc rows repeat one (chain, seq_id); keep the first
            continue
        out[key] = (f[idx["_atom_site.label_comp_id"]],
                    np.array([float(f[idx["_atom_site.Cartn_" + c]]) for c in "xyz"]))
    return out


def gt_rmsd(pred: Path, gt: dict, pairs: list[tuple[int, int]]) -> tuple[float, int]:
    pm = ca_map(pred)
    use = [(a, b) for a, b in pairs if a in pm and b in gt]
    bad = [(a, b) for a, b in use if pm[a][0] != gt[b][0]]
    assert not bad, f"residue identity mismatch against the experimental structure: {bad[:5]}"
    A = np.array([pm[a][1] for a, _ in use])
    B = np.array([gt[b][1] for _, b in use])
    return kabsch_rmsd(A, B), len(use)


def pair_rmsd(p: Path, q: Path) -> float:
    kp, xp = read_atoms(p)
    kq, xq = read_atoms(q)
    assert kp == kq, f"atom identity differs: {p.name} vs {q.name}"
    return kabsch_rmsd(xp, xq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refdir", type=Path, default=CIFS.parent / "ref",
                    help="dir of ref_<size>_seed<n>.cif harvested from the rented GPU")
    ap.add_argument("--gt", type=Path, default=CIFS / "1hcl.cif")
    ap.add_argument("--out", type=Path, default=CIFS.parent / "score.json")
    a = ap.parse_args()

    gt = ca_map(a.gt)
    report: dict = {"gt": {"file": a.gt.name, "resolved_ca": len(gt)}, "sizes": {}}

    for size in (298, 512):
        arms = {arm: sorted(CIFS.glob(f"{size}_{arm}_*.cif")) for arm in ("on", "P")}
        if not arms["on"]:
            continue
        refs = sorted(a.refdir.glob(f"ref_{size}_seed*.cif"),
                      key=lambda p: int(re.search(r"seed(\d+)", p.name).group(1)))
        block: dict = {"arm_cifs": {k: [p.name for p in v] for k, v in arms.items()},
                       "ref_cifs": [p.name for p in refs]}
        print(f"\n===== {size} aa =====")

        # arm vs arm, and the A/A floor: reproduces the campaign's own numbers or the
        # instrument is broken and nothing below it is interpretable.
        block["arm_vs_arm_A"] = round(pair_rmsd(arms["on"][0], arms["P"][0]), 6)
        for arm, cifs in arms.items():
            if len(cifs) > 1:
                block[f"{arm}_self_A"] = round(pair_rmsd(cifs[0], cifs[1]), 6)
        print(f"  on vs P                     {block['arm_vs_arm_A']:10.6f} A")
        for k in ("on_self_A", "P_self_A"):
            if k in block:
                print(f"  {k:27s} {block[k]:10.6f} A  (device A/A floor)")

        # the reference's own spread, which is the scale the margin is read against
        if len(refs) > 1:
            rr = {f"{p.stem[-5:]}|{q.stem[-5:]}": round(pair_rmsd(p, q), 6)
                  for p, q in itertools.combinations(refs, 2)}
            block["ref_vs_ref_A"] = rr
            block["R_max_A"] = max(rr.values())
            block["R_median_A"] = float(np.median(list(rr.values())))
            print(f"  reference seed spread R     min {min(rr.values()):.6f}  "
                  f"median {block['R_median_A']:.6f}  max {block['R_max_A']:.6f} A "
                  f"({len(rr)} pairs)")

        # the number pair the task asks for, per reference seed
        if refs:
            X = {arm: {int(re.search(r"seed(\d+)", r.name).group(1)):
                       round(pair_rmsd(arms[arm][0], r), 6) for r in refs}
                 for arm in ("on", "P")}
            block["X_A"] = X
            for arm in ("on", "P"):
                v = list(X[arm].values())
                block[f"X_{arm}_median_A"] = float(np.median(v))
                block[f"X_{arm}_min_A"] = min(v)
                print(f"  X({arm:2s}, reference)          median {np.median(v):10.6f}  "
                      f"min {min(v):10.6f} A   per-seed {v}")
            block["margin_median_A"] = round(
                block["X_on_median_A"] - block["X_P_median_A"], 6)
            print(f"  margin  X(on) - X(P)        {block['margin_median_A']:+10.6f} A "
                  f"(negative = on closer)")
            if "R_max_A" in block:
                clear = abs(block["margin_median_A"]) > block["R_max_A"]
                block["margin_clears_R"] = bool(clear)
                block["closer_arm"] = "on" if block["margin_median_A"] < 0 else "P"
                print(f"  margin vs R_max             "
                      f"{'CLEARS' if clear else 'INSIDE'} the reference's own spread")

        # the cross-rental / cross-hardware controls, when their CIFs are present
        for name, label in (("512_refH200", "retained H200 2026-08-12"),
                            ("512_refB200", "retained B200 2026-08-12")):
            p = CIFS / f"{name}.cif"
            if size == 512 and p.exists():
                block.setdefault("retained_controls_A", {})
                for arm in ("on", "P"):
                    block["retained_controls_A"][f"{arm}|{name}"] = round(
                        pair_rmsd(arms[arm][0], p), 6)
                if refs:
                    s0 = [r for r in refs if r.name.endswith("seed0.cif")]
                    if s0:
                        block["retained_controls_A"][f"ref_seed0|{name}"] = round(
                            pair_rmsd(s0[0], p), 6)
                print(f"  vs {label}: "
                      + "  ".join(f"{k.split('|')[0]} {v:.6f}"
                                  for k, v in block["retained_controls_A"].items()
                                  if k.endswith(name)))

        # same seed, same code, two different NVIDIA parts: the hardware floor under any
        # arm-to-reference margin, measured rather than assumed
        h2, b2 = CIFS / "512_refH200.cif", CIFS / "512_refB200.cif"
        if size == 512 and h2.exists() and b2.exists():
            block["ref_hw_floor_A"] = round(pair_rmsd(h2, b2), 6)
            print(f"  reference H200 vs B200      {block['ref_hw_floor_A']:10.6f} A "
                  f"(same seed, hardware floor)")

        # the absolute anchor
        block["gt_ca_rmsd_A"] = {}
        for label, pairs in GT_SEGMENTS[size].items():
            row = {}
            for arm in ("on", "P"):
                r, n = gt_rmsd(arms[arm][0], gt, pairs)
                row[arm] = round(r, 6)
            for r_cif in refs:
                v, n = gt_rmsd(r_cif, gt, pairs)
                seed = int(re.search(r"seed(\d+)", r_cif.name).group(1))
                row[f"ref_seed{seed}"] = round(v, 6)
            for name in ("512_refH200", "512_refB200"):
                p = CIFS / f"{name}.cif"
                if size == 512 and p.exists():
                    v, n = gt_rmsd(p, gt, pairs)
                    row[name] = round(v, 6)
            row["n_matched_ca"] = n
            block["gt_ca_rmsd_A"][label] = row
            print(f"  CA RMSD vs 1HCL [{label}] over {n} positions:")
            for k, v in row.items():
                if k != "n_matched_ca":
                    print(f"      {k:16s} {v:10.6f} A")

        report["sizes"][str(size)] = block

    a.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
