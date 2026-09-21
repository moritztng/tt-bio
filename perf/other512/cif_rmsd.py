#!/usr/bin/env python3
"""All-atom Kabsch RMSD between the CIFs each arm of a fold A/B wrote.

Same parser and same superposition as `perf/nontri512/cif_rmsd.py` (commit 26f26116), so the
numbers are comparable across this lineage: Kabsch over ALL atoms, equal weights, which is what
tt-bio's own `kabsch_rmsd` does (memory `tt-bio-kabsch-rmsd-mislabeled-ca-rmsd`: the name says CA,
the function is all-atom).

    cif_rmsd.py <dir-of-arm-dirs> [--ref on]

Each arm dir is `<size>_<arm>_<run>` and holds one CIF. Every pair is reported; the `on` vs `on`
pair is the A/A floor and must be 0.000000 -- anything else means the instrument is broken and no
other number in the run is interpretable.
"""
import argparse
import itertools
import re
from pathlib import Path

import numpy as np


def atom_site_table(p: Path):
    """Parse the mmCIF atom_site loop once. Returns (column -> field position, rows of fields)."""
    lines = p.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j, cols = i + 1, []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                cols.append(lines[j].strip())
                j += 1
            if cols:
                rows = []
                while j < len(lines):
                    s = lines[j].strip()
                    if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                        break
                    f = s.split()
                    if len(f) < len(cols):
                        break
                    rows.append(f)
                    j += 1
                return {c: k for k, c in enumerate(cols)}, rows
            i = j
        else:
            i += 1
    raise SystemExit(f"no _atom_site loop in {p}")


def read_atoms(p: Path):
    """Parse the mmCIF atom_site loop. Returns (keys, coords[N,3])."""
    idx, rows = atom_site_table(p)
    need = ["_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]
    if not all(n in idx for n in need):
        raise SystemExit(f"no coordinates in the _atom_site loop of {p}")
    keycols = [c for c in ("_atom_site.label_asym_id", "_atom_site.label_seq_id",
                           "_atom_site.label_atom_id", "_atom_site.label_comp_id")
               if c in idx]
    keys = [tuple(f[idx[c]] for c in keycols) for f in rows]
    xyz = [[float(f[idx[n]]) for n in need] for f in rows]
    return keys, np.asarray(xyz, dtype=np.float64)


def bfactor_plddt(p: Path):
    """Both readings of the CIF"s own B-factor column, on the 0..1 plDDT scale.

    Which one a model"s reported plDDT equals depends on what it writes per atom. Boltz-2
    writes one plDDT per residue, broadcast to that residue"s atoms, so its reported
    `complex_plddt` is the CA mean; the all-atom mean of the same column is atom-count weighted
    and sits 0.002 higher on cdk2x2_298. Protenix-v2 / OpenFold3 / OpenBind-0 / OpenDDE write a
    genuine per-atom column and report its all-atom mean, 0.025 BELOW the CA mean on the same
    fixture. Reading the wrong one of the two is what made the reported plDDT look inconsistent
    with the column in `perf/k10_p2/FINDINGS.md`. Returns None when there is no B-factor column.
    """
    idx, rows = atom_site_table(p)
    col, name = "_atom_site.B_iso_or_equiv", "_atom_site.label_atom_id"
    if col not in idx or name not in idx:
        return None
    b = [float(f[idx[col]]) for f in rows]
    ca = [v for v, f in zip(b, rows) if f[idx[name]].strip(chr(34)) == "CA"]
    return {"n_atoms": len(b), "n_ca": len(ca),
            "mean_all": round(sum(b) / len(b) / 100.0, 6) if b else None,
            "mean_ca": round(sum(ca) / len(ca) / 100.0, 6) if ca else None}


def mean_ca_bfactor(p: Path):
    """Mean B_iso over CA atoms, on the 0..100 scale the column itself carries."""
    col = bfactor_plddt(p)
    return None if col is None or col["mean_ca"] is None else round(col["mean_ca"] * 100.0, 4)


def plddt_column_check(p: Path, reported, tol: float = 5e-4):
    """Does a reported plDDT equal the mean of the B-factor column the same fold wrote?

    A model writes either one plDDT per atom or one per residue, so exactly one of
    `bfactor_plddt`"s two readings is the number it reports. No model list here on purpose: a
    report path that leaked padding, re-ordered a tensor or picked up a different scalar
    (confidence_score is 0.8*plDDT + 0.2*pTM, so it misses by up to 0.05) matches NEITHER
    reading, which is the regression this catches. `tol` is one count of the writer"s own
    rounding, 3 decimals on the 0..100 scale. `reported` may be on either scale: upstream
    Protenix-v2 and OpenBind report plDDT 0..100 in their own results.json where tt-bio and
    boltz report 0..1, so anything above 1.5 is read as a percentage. A column that is flat zero
    carries no plDDT at all (upstream OpenDDE writes none) and is reported as such rather than as
    a mismatch.
    """
    col = bfactor_plddt(p)
    if col is None or reported is None:
        return {"ok": None, "reason": "no B-factor column" if col is None else "no plddt reported"}
    if col["mean_all"] == 0.0:
        return {"ok": None, "reason": "B-factor column is flat zero, no plDDT written", **col}
    r = round(reported / 100.0, 6) if reported > 1.5 else reported
    gaps = {k: round(r - col[k], 6) for k in ("mean_all", "mean_ca") if col[k] is not None}
    reading = min(gaps, key=lambda k: abs(gaps[k]))
    return {"ok": abs(gaps[reading]) <= tol, "reading": reading, "gap": gaps[reading],
            "reported": r, **col}


def kabsch_rmsd(P, Q):
    """RMSD after optimal rigid superposition. All atoms, equal weights."""
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    P = P @ (V @ np.diag([1.0, 1.0, d]) @ Wt)
    return float(np.sqrt(((P - Q) ** 2).sum() / len(P)))


def arm_of(dirname):
    m = re.match(r"^(\d+)_(.+)_(\d+)$", dirname)
    return (m.group(1), m.group(2)) if m else (None, dirname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--size", default=None)
    ap.add_argument("--ref", default="on")
    a = ap.parse_args()

    data = {}
    for d in sorted(a.root.iterdir()):
        if not d.is_dir():
            continue
        size, arm = arm_of(d.name)
        if a.size and size != a.size:
            continue
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            continue
        k, x = read_atoms(cifs[0])
        data[d.name] = (arm, k, x)
        print(f"  {d.name:26s} {len(x):6d} atoms  {cifs[0].name}")
    if len(data) < 2:
        raise SystemExit("need at least two folds on disk")

    ref_keys = data[sorted(data)[0]][1]
    for name, (_, k, _) in data.items():
        if k != ref_keys:
            raise SystemExit(f"atom identity differs in {name} -- cannot compare by order")
    print(f"\n  atom identity identical across all {len(data)} folds ({len(ref_keys)} atoms), "
          f"so the comparison is atom-for-atom\n")

    print("  pairwise all-atom Kabsch RMSD, A:")
    rows = []
    for x, y in itertools.combinations(sorted(data), 2):
        r = kabsch_rmsd(data[x][2], data[y][2])
        kind = "A/A" if data[x][0] == data[y][0] else "A/B"
        rows.append((kind, x, y, r))
        print(f"    {kind}  {x:26s} vs {y:26s}  {r:11.6f}")

    aa = [r for k, _, _, r in rows if k == "A/A"]
    if aa:
        print(f"\n  A/A structural floor: max {max(aa):.6f} A over {len(aa)} pair(s)")
    print(f"\n  vs the `{a.ref}` arm:")
    refs = [n for n, v in data.items() if v[0] == a.ref]
    for name, (arm, _, x) in sorted(data.items()):
        if arm == a.ref:
            continue
        vals = [kabsch_rmsd(x, data[r][2]) for r in refs]
        if not vals:
            # No arm matched `--ref`, so there is nothing to compare against. A run whose arms are
            # per-SITE tokens (trunk / msa / template) has no arm literally called `on`, and
            # raising here threw away the complete pairwise table already printed above.
            continue
        print(f"    {arm:16s} {min(vals):9.6f} .. {max(vals):9.6f} A")


if __name__ == "__main__":
    main()
