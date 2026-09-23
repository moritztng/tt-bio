"""N(1)-C(last) distance, the ring-closure check, for every structure under the given dirs.

A closed head-to-tail amide sits at 1.33 A; a linear 13-mer's termini are typically > 8 A apart.
Usage: ring.py <dir> ...
"""
import sys
from pathlib import Path

import gemmi

for root in sys.argv[1:]:
    for f in sorted(Path(root).rglob("*.cif")):
        ch = gemmi.read_structure(str(f))[0][0]
        n, c = ch[0].find_atom("N", "*"), ch[len(ch) - 1].find_atom("C", "*")
        d = n.pos.dist(c.pos) if n and c else None
        print(f"{f}  n_res={len(ch)}  N1-C{len(ch)}={d:.2f} A" if d is not None else f"{f} missing atom")
