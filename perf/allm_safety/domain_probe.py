import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/ttuser/.coworker/wt/allm-safety/perf/other512")
import cif_rmsd as CR

OUT = Path("/home/ttuser/.coworker/wt/allm-safety/perf/allm_safety/out")

def load(d):
    p = sorted(d.glob("*.cif"))[0]
    cols, rows = CR.atom_site_table(p)
    xi, yi, zi = (cols["_atom_site.Cartn_x"], cols["_atom_site.Cartn_y"], cols["_atom_site.Cartn_z"])
    si = cols["_atom_site.label_seq_id"]; ai = cols["_atom_site.label_atom_id"]
    xyz, res, nm = [], [], []
    for r in rows:
        xyz.append((float(r[xi]), float(r[yi]), float(r[zi])))
        res.append(int(r[si])); nm.append(r[ai].strip().strip('"'))
    return np.asarray(xyz), np.asarray(res), np.asarray(nm)

def kab(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt((((B @ R.T) - A) ** 2).sum(1).mean()))

pairs = {
    "EFFECT s0 off vs on": ("cif_m18_openfold3_512_s0_leg0_off", "cif_m18_openfold3_512_s0_leg1_on"),
    "SEED   s0 off vs s1 off": ("cif_m18_openfold3_512_s0_leg0_off", "cif_m18_openfold3_512_s1_leg0_off"),
}
X, R, N = load(OUT / "cif_m18_openfold3_512_s0_leg0_off")
print(f"one chain, {len(X)} atoms, residues {R.min()}..{R.max()}")
half = (R.min() + R.max()) // 2
print(f"splitting at residue {half}: N-half {R.min()}..{half}, C-half {half+1}..{R.max()}\n")
print(f"  {'pair':26s} {'global':>9s} {'N-half':>9s} {'C-half':>9s}   (CA-only, A)")
for lab, (d1, d2) in pairs.items():
    A, Ra, Na = load(OUT / d1); B, Rb, Nb = load(OUT / d2)
    assert (Ra == Rb).all() and (Na == Nb).all(), "atom order differs; correspondence unsafe"
    ca = Na == "CA"
    g = kab(A[ca], B[ca])
    n = kab(A[ca & (Ra <= half)], B[ca & (Rb <= half)])
    c = kab(A[ca & (Ra > half)], B[ca & (Rb > half)])
    print(f"  {lab:26s} {g:9.4f} {n:9.4f} {c:9.4f}")
