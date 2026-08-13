"""Kabsch CA-RMSD between two BoltzGen design CIFs, per design index."""
import sys, glob, os
import numpy as np


def ca(path):
    xyz, seq = [], []
    with open(path) as fh:
        cols, inloop, hdr = {}, False, []
        for line in fh:
            if line.startswith("_atom_site."):
                hdr.append(line.strip().split(".")[1]); inloop = True; continue
            if inloop and (line.startswith("ATOM") or line.startswith("HETATM")):
                f = line.split()
                d = dict(zip(hdr, f))
                if d.get("label_atom_id") == "CA":
                    xyz.append([float(d["Cartn_x"]), float(d["Cartn_y"]), float(d["Cartn_z"])])
                    seq.append(d.get("label_comp_id"))
            elif inloop and line.startswith("#"):
                inloop = False
    return np.array(xyz), seq


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, W = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ W))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ W
    Pr = Pc @ R
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean()))


a_dir, b_dir = sys.argv[1], sys.argv[2]
names = sorted(os.path.basename(p) for p in glob.glob(f"{a_dir}/intermediate_designs/*.cif"))
for n in names:
    pa, pb = f"{a_dir}/intermediate_designs/{n}", f"{b_dir}/intermediate_designs/{n}"
    if not os.path.exists(pb):
        print(n, "missing in B"); continue
    A, sa = ca(pa)
    B, sb = ca(pb)
    if A.shape != B.shape:
        print(n, "shape mismatch", A.shape, B.shape); continue
    ident = sum(x == y for x, y in zip(sa, sb)) / max(1, len(sa))
    print(f"{n}: n_CA={len(A)} kabsch_CA_RMSD={kabsch_rmsd(A, B):.4f} A  seq_identity={ident:.4f}")
