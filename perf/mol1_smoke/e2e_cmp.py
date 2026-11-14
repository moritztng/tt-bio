"""Mol-1 TT vs upstream on one protein: CA RMSD of every TT seed to every upstream seed, against the
upstream seed-to-seed floor. Usage: python perf/mol1_smoke/e2e_cmp.py TT_PT UPSTREAM_OUT_DIR SEQ OUT_JSON"""
import glob
import itertools
import json
import statistics
import sys

import torch

from tt_bio.protenix_data import residue_atoms

THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
         "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
         "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def kabsch_rmsd(a, b):
    a, b = a.double() - a.double().mean(0), b.double() - b.double().mean(0)
    u, _, vt = torch.linalg.svd(a.T @ b)
    d = torch.diag(torch.tensor([1.0, 1.0, float(torch.sign(torch.det(u @ vt)))], dtype=torch.float64))
    return float(((a @ (u @ d @ vt)) - b).norm(dim=-1).pow(2).mean().sqrt())


def cif_ca(path):
    rows, cols = [], []
    for line in open(path):
        if line.startswith("_atom_site."):
            cols.append(line.strip().split(".", 1)[1])
        elif line.startswith(("ATOM", "HETATM")) and cols:
            f = line.split()
            r = dict(zip(cols, f))
            if r.get("label_atom_id") == "CA":
                rows.append([float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])])
    return torch.tensor(rows)


tt_pt, up_dir, seq, out = sys.argv[1:5]
idx, off = [], 0
for c in seq:
    names = list(residue_atoms(THREE[c]))
    idx.append(off + names.index("CA"))
    off += len(names)
tt = {int(k): v[idx] for k, v in torch.load(tt_pt, weights_only=False)["coords"].items()}
up = {}
for p in sorted(glob.glob(f"{up_dir}/**/*.cif", recursive=True)):
    sd = int(p.split("seed_")[1].split("/")[0].split("_")[0])
    up.setdefault(sd, cif_ca(p))
assert up and all(len(v) == len(seq) for v in up.values()), {k: len(v) for k, v in up.items()}
floor = [kabsch_rmsd(up[a], up[b]) for a, b in itertools.combinations(sorted(up), 2)]
cross = [kabsch_rmsd(tt[a], up[b]) for a in sorted(tt) for b in sorted(up)]
ttself = [kabsch_rmsd(tt[a], tt[b]) for a, b in itertools.combinations(sorted(tt), 2)]
res = {"n_res": len(seq), "tt_seeds": sorted(tt), "upstream_seeds": sorted(up),
       "upstream_seed_floor_A": {"median": statistics.median(floor), "max": max(floor), "n": len(floor)},
       "tt_vs_upstream_A": {"median": statistics.median(cross), "max": max(cross), "n": len(cross)},
       "tt_seed_spread_A": {"median": statistics.median(ttself), "max": max(ttself), "n": len(ttself)}}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
