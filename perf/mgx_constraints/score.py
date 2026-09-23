"""Score every structure under out/ against what its input asked for.

    score.py [out_dir] [--matrix <mgx_matrix out dir>]  ->  one row per (model, input, sample)

sfti_*          N1-C14 (a closed amide is 1.33 A; 1SFI's is 1.33), SG3-SG11 (1SFI: 2.05),
                CA-RMSD over all 14 residues against 1SFI chain I after Kabsch superposition
cyclic          N1-C13 on the matrix's own cyclic peptide
bond_ligand     the ligand atom nearest SG of A2 and its distance (a C-S bond is 1.82 A)
bond_protein_cys  SG A2 - SG B4 (a disulfide is 2.05 A)
modification    the residue name at position 7 and whether its phosphate P is present
base            max |dxyz| against the same model's base fold in the mgx-matrix run

Thresholds are the matrix harness's (perf/mgx_matrix/analyze.py): a ring is closed below
2.0 A, a bond is present below 2.5 A and a clash at or below 1.0 A. Chains are taken by
position, not by name, because Protenix renames them in input order.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

import gemmi
import numpy as np

HERE = Path(__file__).resolve().parent
STEMS = ("sfti_cyclic_ss", "sfti_cyclic", "sfti_ss", "sfti_linear", "cyclic", "bond_ligand",
         "bond_protein_cys", "modification", "base")


def crystal():
    p = HERE / "1SFI.cif"
    if not p.exists():
        urllib.request.urlretrieve("https://files.rcsb.org/download/1SFI.cif", p)
    ch = gemmi.read_structure(str(p))[0]["I"]
    return np.array([r["CA"][0].pos.tolist() for r in ch if r.name != "HOH"])


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def atom(res, name):
    a = res.find_atom(name, "*")
    return np.array(a.pos.tolist()) if a else None


def dist(x, y):
    return None if x is None or y is None else round(float(np.linalg.norm(x - y)), 2)


def score(stem, path, ref_ca):
    chains = list(gemmi.read_structure(str(path))[0])
    row = {}
    if stem.startswith("sfti") or stem == "cyclic":
        ch = chains[0]
        row["n_c"] = dist(atom(ch[0], "N"), atom(ch[len(ch) - 1], "C"))
        row["ring_closed"] = row["n_c"] is not None and row["n_c"] < 2.0
    if stem.startswith("sfti"):
        ch = chains[0]
        row["sg_sg"] = dist(atom(ch[2], "SG"), atom(ch[10], "SG"))
        row["ca_rmsd"] = round(kabsch_rmsd(np.array([r["CA"][0].pos.tolist() for r in ch]),
                                           ref_ca), 2)
    if stem == "bond_ligand":
        sg = atom(chains[0][1], "SG")
        near = min(((float(np.linalg.norm(np.array(a.pos.tolist()) - sg)), a.name)
                    for c in chains[1:] for r in c for a in r if a.element.name != "H"),
                   default=(None, None))
        row["sg_ligand"], row["nearest"] = (round(near[0], 2) if near[0] else None), near[1]
        row["bonded"] = near[0] is not None and 1.0 < near[0] < 2.5
    if stem == "bond_protein_cys":
        row["sg_sg"] = dist(atom(chains[0][1], "SG"), atom(chains[1][3], "SG"))
    if stem == "modification":
        r7 = chains[0][6]
        row["res7"], row["has_P"] = r7.name, atom(r7, "P") is not None
    return row


def main():
    args = sys.argv[1:]
    matrix = None
    if "--matrix" in args:
        i = args.index("--matrix")
        matrix = Path(args[i + 1])
        del args[i:i + 2]
    out = Path(args[0]) if args else HERE / "out"
    ref_ca = crystal()
    rows = []
    for f in sorted(out.rglob("*.cif")):
        rel = f.relative_to(out).parts
        model, tag = rel[0], rel[1]
        name = f.stem
        stem = next((s for s in STEMS if name == s or name.startswith(s + "_model")), None)
        if stem is None or "msa" in rel or "template" in str(f):
            continue
        sample = int(re.search(r"_model_(\d+)$", name).group(1)) if "_model_" in name else 0
        row = {"model": model, "tag": tag, "input": stem, "sample": sample,
               **score(stem, f, ref_ca)}
        if stem == "base" and matrix:
            ref = next(iter(sorted((matrix / model).rglob("structures/base.cif"))), None)
            if ref is not None:
                a = gemmi.read_structure(str(f))[0]
                b = gemmi.read_structure(str(ref))[0]
                xa = np.array([x.pos.tolist() for c in a for r in c for x in r])
                xb = np.array([x.pos.tolist() for c in b for r in c for x in r])
                row["vs_matrix_max_dxyz"] = (round(float(np.abs(xa - xb).max()), 4)
                                             if xa.shape == xb.shape else f"shape {xa.shape} vs {xb.shape}")
        rows.append(row)
    for r in rows:
        print(json.dumps(r))
    return rows


if __name__ == "__main__":
    main()
