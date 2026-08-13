"""All-atom RMSD between two RFD3 CIFs, raw and after a Kabsch superposition.

Lever A is not bit-exact, so its gate is a trajectory comparison rather than a sha. The reference
scale on this lineage: a seed change is 25.305 A, and RFD3_FAST_GRID -- a lever rejected on accuracy
-- is 6.525 A raw, 6.380 A after Kabsch, coordinate PCC 0.920.

The column indices come from the file's own `loop_` header, not from a count, because the writer's
column order is not part of any contract.
"""
import sys
from pathlib import Path

import torch


def coords(path):
    cols, xs = [], []
    for line in Path(path).read_text().splitlines():
        t = line.strip()
        if t.startswith("_atom_site."):
            cols.append(t.split()[0].split(".", 1)[1])
        elif t.startswith(("ATOM", "HETATM")):
            f = t.split()
            xs.append([float(f[cols.index("Cartn_x")]), float(f[cols.index("Cartn_y")]),
                       float(f[cols.index("Cartn_z")])])
    return torch.tensor(xs, dtype=torch.float64)


def kabsch_rmsd(a, b):
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = torch.linalg.svd(ac.T @ bc)
    d = torch.sign(torch.det(u @ vt))
    r = u @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ vt
    return float(((ac @ r - bc) ** 2).sum(-1).mean().sqrt())


def pcc(a, b):
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


rc = 0
for pa, pb in zip(sys.argv[1::2], sys.argv[2::2]):
    a, b = coords(pa), coords(pb)
    if a.shape != b.shape:
        print(f"{Path(pa).name:16s} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        rc = 1
        continue
    raw = float(((a - b) ** 2).sum(-1).mean().sqrt())
    print(f"{Path(pa).name:16s} atoms {a.shape[0]:5d}   raw RMSD {raw:8.4f} A   "
          f"Kabsch {kabsch_rmsd(a, b):8.4f} A   coord PCC {pcc(a, b):.6f}", flush=True)
raise SystemExit(rc)
