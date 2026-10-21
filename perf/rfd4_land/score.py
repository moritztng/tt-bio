"""Score gate.sh: per fold the top-ranked model's confidences and its accuracy against the crystal,
then per input and arm the spread over seeds 0-7 and a rank test against the 2.2.1 arms.

Usage: python score.py <runs> <1FKG.cif> <aw-go-fkbp8.json> <out.json>

Crystals: 7ROA (examples/ground_truth_structures/prot.cif) for prot and prot_glycan, 1FKG for
fkg_ligand (protein CA superposed, SB3 heavy atoms matched by CCD atom name, no symmetry
correction). TM is a Kabsch-refined lower bound on TM-align. land310 vs land312 at the same seed is
the interpreter effect (same tree, biotite 1.2.0 + backport vs 1.6.0). AICLK is the median of the
2 s sysfs samples taken during the fold.
"""
import json
import sys
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from scipy.stats import mannwhitneyu

ROOT, FKG, AWGO, OUT = (Path(a) for a in sys.argv[1:5])
REPO = Path(__file__).resolve().parents[2]
XTAL = {"prot": REPO / "examples/ground_truth_structures/prot.cif",
        "prot_glycan": REPO / "examples/ground_truth_structures/prot.cif", "fkg_ligand": FKG}
INPUTS, ARMS, SEEDS = ("fkg_ligand", "prot", "prot_glycan"), ("main310", "land310", "land312"), range(8)
BAR_CA, BAR_TM = 3.0, 0.75  # release gate rf3 row, 7ROA vs crystal


def load(p):
    a = pdbx.get_structure(pdbx.CIFFile.read(str(p)), model=1, altloc="first")
    return a[(a.element != "H") & (a.res_name != "HOH")]


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, qc - pc @ R


def rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


def tm(P, Q):
    d0 = 1.24 * np.cbrt(max(len(Q), 19) - 15) - 1.8
    R, t = kabsch(P, Q)
    d = np.linalg.norm(P @ R + t - Q, axis=1)
    best = float((1 / (1 + (d / d0) ** 2)).mean())
    for _ in range(20):
        core = d < max(d0, 4.0)
        if core.sum() < 3:
            break
        R, t = kabsch(P[core], Q[core])
        d = np.linalg.norm(P @ R + t - Q, axis=1)
        best = max(best, float((1 / (1 + (d / d0) ** 2)).mean()))
    return best


def ca_map(a):
    p = a[struc.filter_amino_acids(a) & (a.atom_name == "CA")]
    return {(c, int(r)): x for c, r, x in zip(p.chain_id, p.res_id, p.coord)}


xtal = {}
for inp, p in XTAL.items():
    x = load(p)
    xp = x[struc.filter_amino_acids(x) & (x.atom_name == "CA")]
    xtal[inp] = ({int(r): c for r, c in zip(xp.res_id, xp.coord)},
                 {n: c for n, c in zip(x.atom_name[x.res_name == "SB3"], x.coord[x.res_name == "SB3"])})

folds, top = {}, {}
for inp in INPUTS:
    for arm in ARMS:
        for s in SEEDS:
            run = ROOT / f"{inp}_{arm}_s{s}"
            rc = (run / "rc").read_text().strip() if (run / "rc").exists() else "missing"
            clk = [int(v) for v in (run / "aiclk.txt").read_text().split()] if (run / "aiclk.txt").exists() else []
            row = {"input": inp, "arm": arm, "seed": s, "rc": rc,
                   "aiclk_mhz_median": float(np.median(clk)) if clk else None, "aiclk_samples": len(clk)}
            folds[run.name] = row
            if rc != "0":
                continue
            res = json.loads(next(run.rglob("results.json")).read_text())[0]
            a = load(next(run.rglob(f"structures/{inp}.cif")))
            top[(inp, arm, s)] = a
            xca, xlig = xtal[inp]
            pa = a[struc.filter_amino_acids(a) & (a.atom_name == "CA") & (a.chain_id == "A")]
            common = [int(r) for r in pa.res_id if int(r) in xca]
            P = np.array([pa.coord[pa.res_id == r][0] for r in common]); Q = np.array([xca[r] for r in common])
            R, t = kabsch(P, Q)
            row.update(ranking_score=res["ranking_score"], ptm=res["ptm"], iptm=res.get("iptm"),
                       plddt=res["plddt"], ca_rmsd_xtal=rmsd(P @ R + t, Q), tm_xtal=tm(P, Q), n_ca=len(common))
            if inp == "fkg_ligand":
                la = a[a.res_name == "SB3"]
                names = [n for n in la.atom_name if n in xlig]
                L = np.array([la.coord[la.atom_name == n][0] for n in names]) @ R + t
                row.update(ligand_iptm=res["pair_chains_iptm"]["0"]["1"],
                           lig_rmsd_xtal=rmsd(L, np.array([xlig[n] for n in names])), n_lig_atoms=len(names))


def stats(v):
    v = np.array(v, float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "values_by_seed": [round(float(x), 4) for x in v]}


def mw(a, b):
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) >= 3 and len(b) >= 3 else None


awgo = json.loads(AWGO.read_text())["summary"]["aw221"]
METRICS = {"fkg_ligand": ("ligand_iptm", "lig_rmsd_xtal", "ranking_score", "ca_rmsd_xtal"),
           "prot": ("ca_rmsd_xtal", "tm_xtal", "plddt", "ptm"),
           "prot_glycan": ("ca_rmsd_xtal", "tm_xtal", "plddt", "ptm", "iptm")}
summary, verdict = {}, {}
for inp in INPUTS:
    ok = {arm: [folds[f"{inp}_{arm}_s{s}"] for s in SEEDS if folds[f"{inp}_{arm}_s{s}"]["rc"] == "0"] for arm in ARMS}
    si = {"failed_folds": {arm: 8 - len(ok[arm]) for arm in ARMS}, "arms": {}, "mannwhitney_p": {}, "interpreter_effect": {}}
    for arm in ARMS:
        si["arms"][arm] = {m: stats([r[m] for r in ok[arm]]) for m in METRICS[inp] if ok[arm]}
        if inp == "fkg_ligand" and ok[arm]:
            si["arms"][arm]["pose_within_2A"] = sum(r["lig_rmsd_xtal"] <= 2.0 for r in ok[arm])
    refs = {"main310": None}
    if inp == "fkg_ligand":
        refs["awgo_aw221"] = awgo
    checks = []
    for land in ("land310", "land312"):
        for m in METRICS[inp][:2]:
            lv = [r[m] for r in ok[land]]
            for ref in refs:
                rv = [r[m] for r in ok[ref]] if ref == "main310" else refs[ref][m]["values_by_seed"]
                p = mw(lv, rv)
                si["mannwhitney_p"][f"{land}_vs_{ref}_{m}"] = p
                checks.append(p is not None and p >= 0.05)
        if inp != "fkg_ligand":
            checks += [all(r["ca_rmsd_xtal"] <= BAR_CA and r["tm_xtal"] >= BAR_TM for r in ok[land])]
    for s in SEEDS:
        a, b = top.get((inp, "land310", s)), top.get((inp, "land312", s))
        if a is not None and b is not None:
            ia, ib = ca_map(a), ca_map(b)
            k = sorted(set(ia) & set(ib))
            P, Q = np.array([ia[x] for x in k]), np.array([ib[x] for x in k])
            R, t = kabsch(P, Q)
            si["interpreter_effect"][f"s{s}_ca_rmsd"] = round(rmsd(P @ R + t, Q), 4)
    checks.append(all(v == 0 for v in si["failed_folds"].values()))
    verdict[inp] = "PASS" if all(checks) else "FAIL"
    summary[inp] = si

clk = [f["aiclk_mhz_median"] for f in folds.values() if f["aiclk_mhz_median"]]
out = {"verdict": "PASS" if all(v == "PASS" for v in verdict.values()) else "FAIL", "by_input": verdict,
       "aiclk_mhz_median_of_fold_medians": float(np.median(clk)) if clk else None,
       "aiclk_mhz_fold_median_range": [min(clk), max(clk)] if clk else None,
       "rule": ("PASS per input: 24/24 folds rc=0; for land310 and land312, Mann-Whitney p >= 0.05 against main310 "
                "(and, on fkg_ligand, against aw-go's 8-seed aw221 arm) on the first two metrics listed; on prot and "
                f"prot_glycan every land fold also inside the release bar, CA-RMSD <= {BAR_CA} A and TM >= {BAR_TM}."),
       "summary": summary, "folds": folds}
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("verdict", "by_input", "aiclk_mhz_median_of_fold_medians",
                                        "aiclk_mhz_fold_median_range")}))
for inp in INPUTS:
    for arm in ARMS:
        a = summary[inp]["arms"][arm]
        print(inp, arm, {m: (round(v["median"], 3), round(v["min"], 3), round(v["max"], 3)) for m, v in a.items()
                         if isinstance(v, dict)}, a.get("pose_within_2A", ""))
    print(inp, "p", {k: round(v, 3) for k, v in summary[inp]["mannwhitney_p"].items() if v is not None})
    print(inp, "interp", summary[inp]["interpreter_effect"])
