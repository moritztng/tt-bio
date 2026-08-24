#!/usr/bin/env python3
"""Is the stretch that drives the arm-to-arm RMSD a region the model itself is unsure about?

Reads per-atom pLDDT out of the CIF B_iso_or_equiv column (tt-bio writes plDDT*100 there) and
compares the divergent residues against the rest of the chain, in each arm.
"""
import sys
from pathlib import Path

import numpy as np

LOOP = set(range(157, 168))


def read(p):
    seq, plddt = [], []
    for line in Path(p).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        seq.append(int(f[7]))
        plddt.append(float(f[13]))
    return np.array(seq), np.array(plddt)


for p in sys.argv[1:]:
    seq, pl = read(p)
    inloop = np.isin(seq, list(LOOP))
    print(f"  {Path(p).parent.name:18s} loop 157-167 pLDDT {pl[inloop].mean():6.2f} "
          f"(min {pl[inloop].min():6.2f})   rest of chain {pl[~inloop].mean():6.2f} "
          f"(min {pl[~inloop].min():6.2f})   whole {pl.mean():6.2f}")
