#!/usr/bin/env python3
"""Score two CIFs the way a 2-chain fixture has to be scored.

`cdk2x2_512` is two copies of the same chain. A single all-atom Kabsch RMSD over the whole
complex conflates three different things: per-chain conformation, the rigid-body placement of
one chain against the other, and -- because the copies are identical -- which copy the model
called A. A 12.6 A complex number can mean the fold diverged, or it can mean the two chains
swapped labels and nothing moved. This separates them.

  per-chain      each chain superposed on its own counterpart: conformation only
  complex        whole complex superposed, chains paired A-A/B-B
  complex-swap   whole complex superposed, chains paired A-B/B-A
  plddt          mean |delta| of the B-factor column, which carries no frame at all

Reports the minimum of complex and complex-swap as the symmetry-aware number, which is what
the homodimer's own ambiguity allows.
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch

X, Y, Z, CH, PL = 10, 11, 12, 15, 17


def parse(p):
    per = defaultdict(list)
    plddt = []
    order = []
    for line in Path(p).read_text().splitlines():
        f = line.split()
        if len(f) > 18 and f[0] in ("ATOM", "HETATM"):
            per[f[CH]].append([float(f[X]), float(f[Y]), float(f[Z])])
            plddt.append(float(f[PL]))
            order.append(f[CH])
    return ({c: torch.tensor(v, dtype=torch.float64) for c, v in per.items()},
            torch.tensor(plddt, dtype=torch.float64), order)


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = torch.linalg.svd(Pc.T @ Qc)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=P.dtype))
    return float(torch.sqrt((((Vt.T @ D @ U.T @ Pc.T).T - Qc) ** 2).sum(-1).mean()))


a_ch, a_pl, a_ord = parse(sys.argv[1])
b_ch, b_pl, b_ord = parse(sys.argv[2])
chains = sorted(a_ch)
print("chains %s  atoms %s" % (chains, {c: len(a_ch[c]) for c in chains}))
assert chains == sorted(b_ch) and a_ord == b_ord, "atom ordering differs between the two files"

for c in chains:
    print("per-chain %-3s        %8.4f A" % (c, kabsch_rmsd(a_ch[c], b_ch[c])))

cat = lambda d, order: torch.cat([d[c] for c in order], 0)
direct = kabsch_rmsd(cat(a_ch, chains), cat(b_ch, chains))
print("complex  A-A/B-B     %8.4f A" % direct)
best = direct
if len(chains) == 2:
    swap = kabsch_rmsd(cat(a_ch, chains), cat(b_ch, chains[::-1]))
    print("complex  A-B/B-A     %8.4f A" % swap)
    best = min(direct, swap)
print("symmetry-aware min   %8.4f A" % best)
d = (a_pl - b_pl).abs()
print("plDDT mean|delta| %8.4f   max %8.4f   mean plDDT %.3f / %.3f"
      % (float(d.mean()), float(d.max()), float(a_pl.mean()), float(b_pl.mean())))
