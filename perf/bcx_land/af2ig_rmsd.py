import numpy as np, sys
sys.path.insert(0, "/tmp/bcx-land-tm/scripts/af2_port")
import tap_gate as tg
a = np.load("/tmp/bcx-land-npz/tf_r1.npz"); b = np.load("/tmp/bcx-land-npz/tm_r1.npz")
ref = np.load(tg.ARTIFACTS / "ref_taps.npz")
print([k for k in ref.files if "structure_module#3/final_atom_positions" in k])
def kabsch(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q); d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d]); R = U @ D @ Vt
    return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))
for sm in range(4):
    k = f"structure_module#{sm}/final_atom_positions"
    if k not in a.files: continue
    x, y = a[k].astype(np.float64), b[k].astype(np.float64)
    ca_x, ca_y = x.reshape(-1, 37, 3)[:, 1], y.reshape(-1, 37, 3)[:, 1]
    print(k, x.shape, "CA RMSD floor-vs-main %.4f A" % kabsch(ca_x, ca_y), "max |d| %.4f" % np.abs(x - y).max())
for sm in range(4):
    k = f"structure_module#{sm}/final_atom_positions"
    if k + "/full" not in ref.files: continue
    r = ref[k + "/full"].astype(np.float64).reshape(-1, 37, 3)[:, 1]
    print(k, "CA RMSD vs JAX: floor %.4f A, main %.4f A" % (kabsch(a[k].astype(np.float64).reshape(-1,37,3)[:,1], r),
                                                           kabsch(b[k].astype(np.float64).reshape(-1,37,3)[:,1], r)))
