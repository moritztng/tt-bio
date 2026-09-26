#!/usr/bin/env python3
"""Where do the sub-2.0 A heavy-atom contacts sit in sequence?

The fixture is one protein tiled on a 298-aa period. If the model stacks tandem
copies on top of each other, the clashing pairs cluster at |i-j| = 298*k. If the
fold is torn, they are scattered. This separates those two.

Clash definition copied verbatim from check_structure.py::clashes so the counts
reconcile with the number that instrument reports.
"""
import collections, math, re, sys
import gemmi, numpy as np

CLASH_DIST, DISULFIDE_MAX, PERIOD = 2.0, 2.5, 298
VIRTUAL_ATOM = re.compile(r"V\d+$")

st = gemmi.read_structure(sys.argv[1])
model = st[0]
ns = gemmi.NeighborSearch(st, 5.0).populate()
pairs = {}
for chain in model:
    for res in chain:
        for atom in res:
            if atom.element == gemmi.Element("H") or VIRTUAL_ATOM.match(atom.name):
                continue
            for m in ns.find_atoms(atom.pos, "\0", radius=CLASH_DIST):
                cra = m.to_cra(model)
                if cra.atom.element == gemmi.Element("H") or VIRTUAL_ATOM.match(cra.atom.name):
                    continue
                same_chain = cra.chain.name == chain.name
                if same_chain and abs(cra.residue.seqid.num - res.seqid.num) < 2:
                    continue
                dist = cra.atom.pos.dist(atom.pos)
                if (dist < DISULFIDE_MAX and atom.name == "SG" and cra.atom.name == "SG"
                        and res.name == "CYS" and cra.residue.name == "CYS"):
                    continue
                if dist < CLASH_DIST and cra.atom.serial != atom.serial:
                    key = tuple(sorted((atom.serial, cra.atom.serial)))
                    pairs[key] = (abs(cra.residue.seqid.num - res.seqid.num),
                                  same_chain, dist)

seps = np.array([v[0] for v in pairs.values()])
print(f"{sys.argv[1].split('/')[-1]}: {len(pairs)} clashing heavy-atom pairs")
if not len(seps):
    sys.exit()
off = np.minimum(seps % PERIOD, PERIOD - (seps % PERIOD))
for tol in (0, 5, 15, 30):
    n = int((off <= tol).sum())
    print(f"  |i-j| within +-{tol:2d} of a multiple of {PERIOD}: {n:5d}  ({100*n/len(seps):5.1f} %)")
n_near = int((seps < 30).sum())
print(f"  |i-j| < 30 (local, within one copy)      : {n_near:5d}  ({100*n_near/len(seps):5.1f} %)")
print("  most common |i-j|:", collections.Counter(seps.tolist()).most_common(8))
print(f"  |i-j| median {np.median(seps):.0f}, min {seps.min()}, max {seps.max()}")
