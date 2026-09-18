"""512 aa base-vs-hoist from the post-merge interleaved A/B session's own CIFs.

Five reps, one process, one model load, one device open, arm selected per fold, same seed --
so every rep carries its own paired base/hoist pair AND a base/base A/A pair (p0 and p4 are
both base in every rep). Zero extra device time.
"""
import sys, re, statistics
from pathlib import Path
R = "/home/ttuser/.coworker/wt/c13-land-first"
sys.path.insert(0, R + "/perf/other512")
from cif_rmsd import kabsch_rmsd, read_atoms

d = Path(R + "/perf/c13_land/remeasure512_cifs")
by_rep = {}
for f in d.glob("512_r*_p*_*.cif"):
    m = re.match(r"512_r(-?\d+)_p(\d+)_(\w+)\.cif", f.name)
    by_rep.setdefault(int(m.group(1)), {})[m.group(3) + "#" + m.group(2)] = f

keys, xyz = None, {}
for rep, arms in by_rep.items():
    for tag, f in arms.items():
        k, x = read_atoms(f)
        if keys is None: keys = k
        assert k == keys, f"atom identity differs r{rep} {tag}"
        xyz[(rep, tag)] = x
ca = [i for i, k in enumerate(keys) if k[2] == "CA"]

lever, aa = [], []
for rep in sorted(by_rep):
    a = [t for t in by_rep[rep] if t.startswith("base#")]
    h = [t for t in by_rep[rep] if t.startswith("hoist#")]
    if len(a) >= 1 and h:
        v = kabsch_rmsd(xyz[(rep, a[0])], xyz[(rep, h[0])])
        vca = kabsch_rmsd(xyz[(rep, a[0])][ca], xyz[(rep, h[0])][ca])
        lever.append((rep, v, vca))
    if len(a) >= 2:
        aa.append((rep, kabsch_rmsd(xyz[(rep, a[0])], xyz[(rep, a[1])])))

print("512 aa cdk2x2, base vs hoist (THE SHIPPED CONFIGURATION), one rep = one paired fold pair")
for rep, v, vca in sorted(lever):
    print(f"   rep {rep:>2}   {v:8.5f} A all-atom / {vca:8.5f} A CA")
vals = [v for _, v, _ in lever]
print(f"   worst {max(vals):.5f} A, mean {statistics.mean(vals):.5f} A, n={len(vals)}")
print("A/A control, base vs base inside the same rep (same seed, same process):")
for rep, v in sorted(aa):
    print(f"   rep {rep:>2}   {v:8.5f} A all-atom")
print(f"   worst A/A {max(v for _, v in aa):.5f} A, n={len(aa)}")
