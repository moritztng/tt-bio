#!/usr/bin/env python3
"""Is a coordinate difference between the arms actually a difference between the arms?

Round 3 of the paired sweep came back array_equal=False on protenix-v2 at 2.1385 A, against five
other bit-exact comparisons. It is not C2FIX. `BASE_protenix-v2_298_r3` reports
intra_run_max_abs_delta_A = 2.1385: the two warm folds inside that ONE BASE process disagree with
each other by exactly the amount the arms appear to disagree by. Every other run on both arms is
0.0, C2FIX is identical across all three of its rounds, and BASE r1 == r2, so C2FIX still equals
the BASE majority and main is the arm that flaked.

Two views, both card-free, both needed:
  - the intra_run delta fold_ab already records, which is what makes this diagnosable at all;
  - every same-arm cross-round pair, which answers "is this fold deterministic" without reference
    to the other arm.

    python3 perf/w6_c2fix/determinism.py
"""

import glob, json, itertools
from pathlib import Path
import numpy as np

OUT = Path("perf/w6_c2fix/out")

print("=== intra-run max abs delta reported by fold_ab (warm folds within one process) ===")
for f in sorted(glob.glob(str(OUT / "*_298_r*.json"))):
    d = json.load(open(f))
    k = [x for x in d if "intra" in x.lower()]
    print(f"  {Path(f).stem:34s} " + ", ".join(f"{x}={d[x]}" for x in k) + f"   warm={[round(w,2) for w in d['warm_s']]}")

print()
print("=== cross-ROUND comparison within the SAME arm (is the fold deterministic at all?) ===")
for m in ("protenix-v2", "opendde"):
    for arm in ("BASE", "C2FIX"):
        dirs = sorted(OUT.glob(f"{arm}_{m}_298_r*"))
        dirs = [d for d in dirs if d.is_dir() and (d / "coords.npy").is_file()]
        arrs = {d.name[-2:]: np.load(d / "coords.npy") for d in dirs}
        for a, b in itertools.combinations(sorted(arrs), 2):
            A, B = arrs[a], arrs[b]
            print(f"  {m:12s} {arm:6s} {a} vs {b}: array_equal={np.array_equal(A, B)}"
                  f"  max abs delta {np.abs(A - B).max():.3e} A")
