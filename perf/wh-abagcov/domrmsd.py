"""Global CA-RMSD over a TILED chimera is dominated by how the copies are packed against each
other, and that packing has no native answer to be right about. So report both: the global
number and the per-copy number, each copy superposed on its own."""
import sys, gemmi, numpy as np
def ca(p):
    st = gemmi.read_structure(p); st.setup_entities()
    return np.array([[a.pos.x, a.pos.y, a.pos.z]
                     for ch in st[0] for r in ch for a in [r.find_atom("CA", "*")] if a])
def rms(A, B):
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return float(np.sqrt(((Ac - (R @ Bc.T).T) ** 2).sum() / len(A)))
A, B = ca(sys.argv[1]), ca(sys.argv[2])
n = min(len(A), len(B)); A, B = A[:n], B[:n]
P = int(sys.argv[3]) if len(sys.argv) > 3 else 298
print(f"CA atoms={n}  GLOBAL Kabsch RMSD={rms(A,B):.4f} A")
for i in range(0, n, P):
    a, b = A[i:i+P], B[i:i+P]
    if len(a) < 30: continue
    print(f"  copy {i:4d}-{i+len(a)-1:4d} ({len(a)} CA): {rms(a,b):.4f} A")
