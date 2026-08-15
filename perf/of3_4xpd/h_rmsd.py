#!/usr/bin/env python3
"""All-atom Kabsch RMSD between lever H's arms, with both A/A floors.

Same parser and same superposition as `perf/other512/cif_rmsd.py`, so the numbers are
comparable across this lineage: Kabsch over ALL atoms, equal weights. Takes the flat CIF
directory `perf/of3_4xpd/h_ab.py --cifdir` writes.
"""
import itertools, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
from cif_rmsd import read_atoms, kabsch_rmsd  # noqa: E402

d = Path(sys.argv[1])
cifs = sorted(p for p in d.iterdir() if p.suffix == ".cif" and not p.name.startswith("cold"))
data = {}
for p in cifs:
    k, x = read_atoms(p)
    data[p.name] = ("memo" if "_memo_" in p.name else "plain", k, x)
    print(f"  {p.name:40s} {len(x):6d} atoms")

ref = data[cifs[0].name][1]
for n, (_, k, _) in data.items():
    if k != ref:
        raise SystemExit(f"atom identity differs in {n} -- cannot compare atom-for-atom")
print(f"\n  atom identity identical across {len(data)} folds ({len(ref)} atoms)\n")

print("  pairwise all-atom Kabsch RMSD, A:")
for x, y in itertools.combinations(sorted(data), 2):
    kind = "A/A" if data[x][0] == data[y][0] else "A/B"
    print(f"    {kind}  {x:40s} vs {y:40s}  {kabsch_rmsd(data[x][2], data[y][2]):11.6f}")
