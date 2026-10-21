"""Score ab8.sh: per fold the top-ranked model's confidences, protein CA-RMSD and SB3 heavy-atom RMSD
against the 1FKG crystal (superposed on protein CA, ligand atoms matched by CCD atom name), then the
per-arm spread and a rank test between arms. Usage: python score8.py <runs> <crystal.cif> <out.json>"""
import json
import sys
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from scipy.stats import mannwhitneyu

ROOT, XTAL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ARMS = ("aw221", "aw304", "aw304rot")


def load(p):
    a = pdbx.get_structure(pdbx.CIFFile.read(str(p)), model=1, altloc="first")
    return a[(a.element != "H") & (a.res_name != "HOH")]


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return R, qc - pc @ R


def rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


x = load(XTAL)
x_ca = {int(r): c for r, c in zip(x.res_id, x.coord) for _ in [0]} if False else None
xp = x[struc.filter_amino_acids(x) & (x.atom_name == "CA")]
xca = {int(r): c for r, c in zip(xp.res_id, xp.coord)}
xl = x[x.res_name == "SB3"]
xlig = {n: c for n, c in zip(xl.atom_name, xl.coord)}

rows = {}
for run in sorted(ROOT.glob("aw*_s*")):
    if "." in run.name:
        continue
    arm, seed = run.name.rsplit("_s", 1)
    rc = (run / "rc").read_text().strip() if (run / "rc").exists() else "missing"
    if rc != "0":
        rows[run.name] = {"arm": arm, "seed": int(seed), "rc": rc}
        continue
    res = json.loads(next(run.rglob("results.json")).read_text())[0]
    a = load(next(run.rglob("fkg_ligand.cif")))
    pa = a[struc.filter_amino_acids(a) & (a.atom_name == "CA")]
    common = [r for r in pa.res_id if int(r) in xca]
    P = np.array([pa.coord[pa.res_id == r][0] for r in common]); Q = np.array([xca[int(r)] for r in common])
    R, t = kabsch(P, Q)
    la = a[a.res_name == "SB3"]
    names = [n for n in la.atom_name if n in xlig]
    L = np.array([la.coord[la.atom_name == n][0] for n in names]) @ R + t
    LX = np.array([xlig[n] for n in names])
    rows[run.name] = {
        "arm": arm, "seed": int(seed), "rc": "0",
        "ranking_score": res["ranking_score"], "iptm": res["iptm"], "ptm": res["ptm"], "plddt": res["plddt"],
        "ligand_iptm": res["pair_chains_iptm"]["0"]["1"],
        "ca_rmsd_xtal": rmsd(P @ R + t, Q), "lig_rmsd_xtal": rmsd(L, LX), "n_lig_atoms": len(names), "n_ca": len(common),
    }

summary = {}
METRICS = ("ligand_iptm", "iptm", "ranking_score", "lig_rmsd_xtal", "ca_rmsd_xtal")
for arm in ARMS:
    rs = [r for r in rows.values() if r["arm"] == arm and r["rc"] == "0"]
    s = {"n": len(rs), "failed_folds": sum(r["arm"] == arm and r["rc"] != "0" for r in rows.values())}
    for m in METRICS:
        v = np.array([r[m] for r in rs])
        if len(v):
            s[m] = {"median": float(np.median(v)), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                    "values_by_seed": [round(float(r[m]), 4) for r in sorted(rs, key=lambda r: r["seed"])]}
    if rs:
        s["pose_within_2A"] = sum(r["lig_rmsd_xtal"] <= 2.0 for r in rs)
    summary[arm] = s
tests = {}
for a, b in (("aw221", "aw304"), ("aw221", "aw304rot"), ("aw304", "aw304rot")):
    for m in ("ligand_iptm", "lig_rmsd_xtal", "ranking_score"):
        va = [r[m] for r in rows.values() if r["arm"] == a and r["rc"] == "0"]
        vb = [r[m] for r in rows.values() if r["arm"] == b and r["rc"] == "0"]
        if len(va) >= 3 and len(vb) >= 3:
            tests[f"{a}_vs_{b}_{m}"] = float(mannwhitneyu(va, vb, alternative="two-sided").pvalue)
out = {"crystal": XTAL.name, "summary": summary, "mannwhitney_p": tests, "folds": rows,
       "note": "top-ranked of 5 samples per fold; ligand RMSD after superposing protein CA onto the crystal, "
               "SB3 atoms matched by CCD name, no symmetry correction (same for every arm). Accuracy only, no AICLK."}
OUT.write_text(json.dumps(out, indent=1))
for arm, s in summary.items():
    print(arm, "n", s["n"], "within2A", s.get("pose_within_2A"),
          {m: (round(s[m]["median"], 3), round(s[m]["min"], 3), round(s[m]["max"], 3)) for m in METRICS if m in s})
print(json.dumps(tests, indent=0))
