#!/usr/bin/env python3
"""Does the dividing-k lever move a RIGID part of the structure, or only the hinges?

The 832-token fixture is a tiled CDK2 repeat, so a whole-structure CA-RMSD is dominated by how
the copies happen to be arranged relative to each other, which re-seeding rearranges freely. That
gave a 17-21 A seed floor and a test with no power. Restricting the superposition and the score
to ONE copy removes the hinge degree of freedom without refolding anything: inside a copy the
fold is the same protein every time, so whatever floor remains is real disagreement.

Reports, per window: within-arm pairs (seed only) against across-arm pairs (seed + lever), and an
exact permutation over the arm labels.
"""
import itertools
import json
import sys

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx


def cas(path):
    a = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    a = a[struc.filter_amino_acids(a) & (a.atom_name == "CA") & (a.element != "H")]
    return a


def rmsd_window(p, q, lo, hi):
    a, b = cas(p)[lo:hi], cas(q)[lo:hi]
    assert a.array_length() == b.array_length() == hi - lo, (a.array_length(), b.array_length())
    fitted, _ = struc.superimpose(a, b)
    return float(struc.rmsd(a, fitted))


def mean(xs):
    return sum(xs) / len(xs)


RUNS = json.loads(sys.argv[1])
names = sorted(RUNS)
arm = {n: n.split(":")[0] for n in names}
out = {}
for label, (lo, hi) in {"full 1-832": (0, 832), "copy1 1-298": (0, 298),
                        "copy2 299-596": (298, 596), "core 1-150": (0, 150)}.items():
    d = {(a, b): rmsd_window(RUNS[a], RUNS[b], lo, hi)
         for a, b in itertools.combinations(names, 2)}
    within = [v for (a, b), v in d.items() if arm[a] == arm[b]]
    across = [v for (a, b), v in d.items() if arm[a] != arm[b]]
    obs = mean(across) - mean(within)
    n_on = sum(1 for n in names if arm[n] == "on")
    cnt = tot = 0
    for on in itertools.combinations(names, n_on):
        lab = {n: ("on" if n in on else "off") for n in names}
        w = [v for (a, b), v in d.items() if lab[a] == lab[b]]
        x = [v for (a, b), v in d.items() if lab[a] != lab[b]]
        if not w or not x:
            continue
        tot += 1
        cnt += abs(mean(x) - mean(w)) >= abs(obs) - 1e-12
    out[label] = {"within_mean": mean(within), "within_max": max(within),
                  "across_mean": mean(across), "across_min": min(across),
                  "across_max": max(across), "sep": obs, "p": cnt / tot,
                  "pairs": {f"{a}|{b}": v for (a, b), v in d.items()}}
    print(f"{label:16s} within {mean(within):7.3f} (max {max(within):6.3f})   "
          f"across {mean(across):7.3f} ({min(across):6.3f}..{max(across):6.3f})   "
          f"sep {obs:+7.3f}  p={cnt / tot:.2f}")

with open("/home/ttuser/.coworker/wt/land-standing/perf/land_standing/out/of3_832/"
          "domain_rmsd.json", "w") as fh:
    json.dump({"runs": RUNS, "windows": out}, fh, indent=1)
print("wrote domain_rmsd.json")
