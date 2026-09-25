"""Kabsch CA RMSD between two CIFs of the same sequence, float64."""
import sys
import numpy as np
import gemmi

def ca(p):
    s = gemmi.read_structure(p); s.setup_entities()
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for ch in s[0] for r in ch for a in r if a.name == "CA"], dtype=np.float64)

a, b = ca(sys.argv[1]), ca(sys.argv[2])
assert a.shape == b.shape, (a.shape, b.shape)
a -= a.mean(0); b -= b.mean(0)
u, _, vt = np.linalg.svd(a.T @ b)
d = np.sign(np.linalg.det(u @ vt))
r = u @ np.diag([1, 1, d]) @ vt
print(f"n_ca={len(a)} rmsd_A={np.sqrt(((a @ r - b) ** 2).sum(1).mean()):.6f} max_dev_A={np.sqrt(((a @ r - b) ** 2).sum(1)).max():.4f}")
