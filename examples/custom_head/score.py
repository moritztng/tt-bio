"""Score ContactHead's map and the fold's own structure against the true contacts.

    python score.py STRUCTURES TRUTH

Precision of the top L/5 and top L predicted pairs, the usual contact-map measure, at
sequence separation 12+ and 24+. The structure's prediction ranks pairs by its own Cb-Cb
distance; the head ranks them by its probability.
"""

import sys
from pathlib import Path

import gemmi
import numpy as np


def structure_scores(cif):
    chain = gemmi.read_structure(str(cif))[0][0]
    xyz = np.array([(r.find_atom("CB", "*") or r.find_atom("CA", "*")).pos.tolist() for r in chain])
    return -np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)


def precision(scores, truth, min_sep, top):
    n = len(truth)
    i, j = np.triu_indices(n, k=min_sep)
    order = np.argsort(-scores[i, j])[:top]
    return float(truth[i[order], j[order]].mean())


def main(structures, truth):
    for f in sorted(Path(structures).glob("*_ContactHead.npz")):
        rec = f.name.removesuffix("_ContactHead.npz")
        y = np.load(Path(truth) / f"{rec}.npy")
        n = len(y)
        maps = {"ContactHead": np.load(f)["contact_probs"],
                "fold structure": structure_scores(Path(structures) / f"{rec}.cif")}
        print(f"{rec}: {n} residues")
        for sep in (12, 24):
            i, j = np.triu_indices(n, k=sep)
            print(f"  |i-j| >= {sep}: base rate {y[i, j].mean():.3f}")
            for name, m in maps.items():
                print(f"    {name:15s} top-L/5 {precision(m, y, sep, n // 5):.3f}"
                      f"   top-L {precision(m, y, sep, n):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
