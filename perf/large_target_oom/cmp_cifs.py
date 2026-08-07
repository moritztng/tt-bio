#!/usr/bin/env python3
"""Fold-level A/B: CA-Kabsch RMSD + pLDDT PCC between two mmCIF structure files.

Used for the large-target OOM workstream's branch-vs-main equivalence leg: same target,
same seed, same config, so any delta is the code change alone. CA rows are keyed on
(chain, resseq) so a chain-ordering difference cannot fake a coordinate one.
"""
import sys

import numpy as np


def parse_ca(path):
    coords, plddt, keys = [], [], []
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        t = line.split()
        if len(t) < 18 or t[2] != "CA":
            continue
        keys.append((t[5], int(t[7])))
        plddt.append(float(t[13]))
        coords.append([float(t[15]), float(t[16]), float(t[17])])
    return keys, np.array(coords), np.array(plddt)


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    H = a.T @ b
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    rot = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    dev = np.sqrt(((a @ rot.T - b) ** 2).sum(-1))
    return float(dev.mean()), float(dev.max())


ka, ca, pa = parse_ca(sys.argv[1])
kb, cb, pb = parse_ca(sys.argv[2])
common = [k for k in ka if k in set(kb)]
ia = [ka.index(k) for k in common]
ib = [kb.index(k) for k in common]
rmsd, maxdev = kabsch_rmsd(ca[ia], cb[ib])
pcc = float(np.corrcoef(pa[ia], pb[ib])[0, 1])
print(f"CA atoms compared: {len(common)} (a={len(ka)} b={len(kb)})")
print(f"CA kabsch RMSD: {rmsd:.4f} A  (max per-residue {maxdev:.4f} A)")
print(f"pLDDT: mean a={pa.mean():.2f} b={pb.mean():.2f}  PCC={pcc:.6f}")
