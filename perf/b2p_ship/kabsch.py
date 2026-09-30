"""Superposed CA-RMSD (Kabsch) between two mmCIFs of the same fold, plus mean pLDDT from each."""
import sys
import numpy as np


def ca_and_plddt(path):
    cols, xyz, b = None, [], []
    for line in open(path):
        if line.startswith("_atom_site."):
            cols = cols or []
            cols.append(line.strip().split(".")[1])
            continue
        if cols and line.startswith(("ATOM", "HETATM")):
            f = line.split()
            if len(f) < len(cols):
                continue
            r = dict(zip(cols, f))
            if r.get("label_atom_id") == "CA":
                xyz.append([float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])])
                try:
                    b.append(float(r.get("B_iso_or_equiv", "nan")))
                except ValueError:
                    b.append(float("nan"))
        elif cols and line.startswith("#") and xyz:
            break
    return np.array(xyz), np.array(b)


A, pa = ca_and_plddt(sys.argv[1])
B, pb = ca_and_plddt(sys.argv[2])
n = min(len(A), len(B))
A, B = A[:n], B[:n]
print(f"CA {n}  mean_plddt_arm1 {np.nanmean(pa):.2f}  mean_plddt_arm2 {np.nanmean(pb):.2f}")
print(f"RMSD_NOSUPERPOSE {np.sqrt(((A - B) ** 2).sum(1).mean()):.4f} A")
Ac, Bc = A - A.mean(0), B - B.mean(0)
U, S, Vt = np.linalg.svd(Ac.T @ Bc)
d = np.sign(np.linalg.det(U @ Vt))
R = U @ np.diag([1, 1, d]) @ Vt
print(f"RMSD_SUPERPOSED {np.sqrt(((Ac @ R - Bc) ** 2).sum(1).mean()):.4f} A")
r = np.sqrt((Ac ** 2).sum(1).mean())
print(f"RADIUS_OF_GYRATION_arm1 {r:.2f} A  (a deviation near this is 'unrelated fold')")
