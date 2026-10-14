"""Accuracy table for one model's arms: every arm against `off` on the same fold (CA Kabsch RMSD, raw max |dx|), the
det pair (exact-det vs off-det: max |dx| 0 means bitwise), and the `off` run-to-run floor (fold i vs fold 0, same input).

    python3 perf/kitcmp/acc_arms.py <results>/<model>     -> markdown rows: pair | folds | rmsd | per-copy rmsd | max_dx
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from acc import ca, kabsch  # noqa: E402


def structures(arm):
    """fold name -> CA coords of that fold's first model (fold0 = the cold fold)."""
    out = {}
    for p in sorted(arm.rglob("*.cif")):
        m = re.search(r"(fold\d+)", p.name)
        if m and m.group(1) not in out:
            out[m.group(1)] = ca(p)[0]
    return out


def compare(a, b):
    folds = sorted(set(a) & set(b))
    r = [kabsch(a[f], b[f]) for f in folds]
    h = {f: len(a[f]) // 2 for f in folds}  # two CDK2 copies in one chain: the worse copy, without the copy placement
    rh = [max(kabsch(a[f][:h[f]], b[f][:h[f]]), kabsch(a[f][h[f]:], b[f][h[f]:])) for f in folds]
    dx = [float(np.abs(a[f] - b[f]).max()) for f in folds]
    return len(folds), r, rh, dx


def row(name, n, r, rh, dx):
    if not n:
        return f"| {name} | 0 | - | - | - |"
    return f"| {name} | {n} | {np.median(r):.3f} ({max(r):.3f}) | {np.median(rh):.3f} ({max(rh):.3f}) | {max(dx):.4f} |"


def main(d):
    d = Path(d)
    arms = {p.name: structures(p) for p in sorted(d.iterdir()) if p.is_dir()}
    arms = {k: v for k, v in arms.items() if v}
    off = arms.get("off", {})
    print(f"{d.name}: pair | folds | CA RMSD A median (max) | per-copy CA RMSD A median (max) | max |dx| A")
    for k in ("exact", "fast"):
        if k in arms and off:
            print(row(f"{k} vs off", *compare(arms[k], off)))
    if "exact-det" in arms and "off-det" in arms:
        print(row("exact-det vs off-det", *compare(arms["exact-det"], arms["off-det"])))
    if "fold0" in off:
        rest = {f: v for f, v in off.items() if f != "fold0"}
        print(row("off fold i vs fold 0 (floor)", *compare(rest, {f: off["fold0"] for f in rest})))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
