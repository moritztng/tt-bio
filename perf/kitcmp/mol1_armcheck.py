"""Quality bar for a MOL1 B200 arm (state/mol1/PROTOCOL.md section 6): does the candidate arm's ranked-top structure
agree with the default `cueq-cueq` arm within the seed floor?

    python mol1_armcheck.py <session dir> <candidate arm> [sizes...]

Per size and seed, the top-ranked sample of each arm is aligned on C-alpha (Kabsch). Arm deviation = candidate vs
default at the same seed; seed floor = default seed s vs default seed s+1. Ligand RMSD is taken in the C-alpha frame.
Prints one JSON line per size with medians and the KS p-value of arm deviations against the seed floor.
"""
import json, sys
from pathlib import Path

import gemmi
import numpy as np
from scipy.stats import ks_2samp


def top(seed_dir):
    best = None
    for f in seed_dir.glob("predictions/*_summary_confidence_sample_*.json"):
        d = json.loads(f.read_text())
        s = d.get("ranking_score", next((v for k, v in d.items() if "ranking" in k), 0.0))
        i = f.stem.rsplit("_", 1)[1]
        if best is None or s > best[0]:
            best = (s, next(seed_dir.glob(f"predictions/*_sample_{i}.cif")))
    st = gemmi.read_structure(str(best[1])); st.setup_entities()
    ca, lig = [], []
    for ch in st[0]:
        for r in ch:
            if r.het_flag == "H" or gemmi.find_tabulated_residue(r.name) is None or not gemmi.find_tabulated_residue(r.name).is_amino_acid():
                if r.het_flag == "H":
                    lig += [a.pos.tolist() for a in r if a.element.name != "H"]
                continue
            a = r.find_atom("CA", "*")
            if a: ca.append(a.pos.tolist())
    return np.array(ca), np.array(lig).reshape(-1, 3)


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return lambda X: (X - pc) @ R + qc


def dev(a, b):
    f = kabsch(a[0], b[0])
    ca = float(np.sqrt(((f(a[0]) - b[0]) ** 2).sum(1).mean()))
    lig = float(np.sqrt(((f(a[1]) - b[1]) ** 2).sum(1).mean())) if len(a[1]) and len(a[1]) == len(b[1]) else None
    return ca, lig


def cell(sess, size, arm):
    ds = sorted(sess.glob(f"{size}-{arm}-k1/r*/p0/*/seed_*"))
    return {int(d.name.split("_")[1]): top(d) for d in ds}


def main():
    sess, cand = Path(sys.argv[1]), sys.argv[2]
    for size in sys.argv[3:] or ["256", "512", "1024"]:
        base, arm = cell(sess, size, "cueq-cueq"), cell(sess, size, cand)
        seeds = sorted(set(base) & set(arm))
        nan = any(not np.isfinite(x[0]).all() for x in list(base.values()) + list(arm.values()))
        a = [dev(arm[s], base[s]) for s in seeds]
        f = [dev(base[s], base[t]) for s, t in zip(seeds, seeds[1:])]
        out = {"size": int(size), "arm": cand, "seeds": seeds, "nan": nan,
               "arm_ca_rmsd": [round(x[0], 3) for x in a], "floor_ca_rmsd": [round(x[0], 3) for x in f],
               "arm_lig_rmsd": [None if x[1] is None else round(x[1], 3) for x in a],
               "floor_lig_rmsd": [None if x[1] is None else round(x[1], 3) for x in f],
               "ks_p_ca": round(float(ks_2samp([x[0] for x in a], [x[0] for x in f]).pvalue), 4)}
        la, lf = [x[1] for x in a if x[1] is not None], [x[1] for x in f if x[1] is not None]
        if la and lf:
            out["ks_p_lig"] = round(float(ks_2samp(la, lf).pvalue), 4)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
