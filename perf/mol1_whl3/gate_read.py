"""Read a merge-gate cell (out/gate-<model>-512): per arm the warm fold walls, the AICLK of the folding node over the arm's
window, pLDDT, and the CA RMSD of each fix structure to the base structure of the same fold tag.
    python3 gate_read.py CELL_DIR"""
import json, statistics, sys
from datetime import datetime
from pathlib import Path
import numpy as np

d = Path(sys.argv[1])
node = Path(d / "nodes.txt").read_text().split()[0].split("/")[-1]
clk = [l.split("\t") for l in (d / "aiclk.tsv").read_text().splitlines()]
clk = [(float(t), int(c)) for t, n, c, _ in (x for x in clk if len(x) == 4) if n == node and c.strip().isdigit()]
arms = {a: json.loads((d / a / "result.json").read_text()) for a in ("base", "fix", "base2") if (d / a / "result.json").exists()}
starts = {a: datetime.strptime(r["started"], "%Y-%m-%dT%H:%M:%S%z").timestamp() for a, r in arms.items()}
order = sorted(starts, key=starts.get)

def ca(p):
    xyz = []
    for l in p.read_text().splitlines():
        f = l.split()
        if f and f[0] == "ATOM" and f[2] == "CA":   # _atom_site: Cartn_x/y/z are fields 15-17
            xyz.append([float(f[15]), float(f[16]), float(f[17])])
    return np.array(xyz)

def rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    dd = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, dd]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))

print(f"node {node}")
for i, a in enumerate(order):
    end = starts[order[i + 1]] if i + 1 < len(order) else float("inf")
    c = [v for t, v in clk if starts[a] <= t < end]
    warm = [f["fold_s"] for f in arms[a]["folds"] if not f["tag"].startswith("cold")]
    pl = [f["plddt"] for f in arms[a]["folds"]]
    print(f"{a:6s} warm {warm} median {statistics.median(warm) if warm else None}  plddt {pl}  "
          f"AICLK node {node}: median {statistics.median(c) if c else None} min {min(c) if c else None} n {len(c)}")
for a in ("fix", "base2"):
    for p in sorted((d / a).glob("*.cif")):
        q = d / "base" / p.name
        if q.exists():
            x, y = ca(p), ca(q)
            if len(x) == len(y) and len(x):
                print(f"{a} vs base {p.name}: CA RMSD {rmsd(x, y):.3f} A ({len(x)} CA)")
