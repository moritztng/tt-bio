"""BOARD-DET: compare two trunk_step --grad-dump npz files tensor by tensor (bit-equal count, worst rel L2, first names)."""
import sys
import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ka, kb = set(a.files), set(b.files)
if ka != kb:
    print("KEY MISMATCH", sorted(ka ^ kb)[:10]); sys.exit(2)
rows = []
for k in sorted(ka):
    x, y = a[k].astype(np.float64), b[k].astype(np.float64)
    if np.array_equal(a[k], b[k]):
        continue
    rows.append((float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30)), k))
rows.sort(reverse=True)
print(f"tensors {len(ka)}, bit-identical {len(ka) - len(rows)}, differing {len(rows)}")
for r, k in rows[:15]:
    print(f"  {r:.3e}  {k}")
sys.exit(1 if rows else 0)
