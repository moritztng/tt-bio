#!/usr/bin/env python3
"""Does a refolded complex actually have an interface?

Written for bcw-accept's REFOLD finding. The one arm B trajectory that finished the
design ladder at 800 tokens passed every stage at i_pTM 0.86 and refolded to i_pTM 0.20
with Interface_Residues 0. Before that could be written up as a design finding it had to
be separated from a harness defect: a refold that truncated the 614-residue target or
mishandled the chains would report Interface_Residues 0 as an artifact, and that would be
an engine bug rather than anything about the token axis.

So this reads the structures rather than the score table. It prints, per complex:
  - the per-chain CA count, so truncation is visible;
  - the minimum CA-CA distance between the two chains, which is what a backbone
    interface criterion sees;
  - the minimum heavy-atom distance, which separates "docked and grazing" from
    "flung off into solvent".

A binder with min heavy-atom near 3-4 A and min CA-CA above 8 A is touching the target
with side chains while its backbone never closes. That is a real structure, not a
parsing failure, and it is what arm B produced at 800.

Usage:
    python3 perf/bcw_accept/refold_iface.py perf/bcw_accept/out/armB1
    python3 perf/bcw_accept/refold_iface.py <dir-of-cifs> --ca-cutoff 8.0
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

# mmCIF _atom_site column order as BindCraft 2 writes it. Verified against
# armB1/2_Refolded/Complexes/*.cif rather than assumed: an earlier guess at these
# indices silently parsed zero atoms and reported every complex as having no chains.
ATOM_NAME = 2
CHAIN = 11
X, Y, Z = 14, 15, 16


def read_chains(path):
    """-> {chain: {"ca": [xyz], "all": [xyz]}}, parsed off auth_asym_id."""
    ca = defaultdict(list)
    allatom = defaultdict(list)
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        if len(f) <= Z:
            continue
        try:
            xyz = (float(f[X]), float(f[Y]), float(f[Z]))
        except ValueError:
            continue
        chain = f[CHAIN]
        allatom[chain].append(xyz)
        if f[ATOM_NAME] == "CA":
            ca[chain].append(xyz)
    return {c: {"ca": np.array(ca[c]), "all": np.array(allatom[c])} for c in allatom}


def min_dist(a, b, chunk=2000):
    """Minimum pairwise distance, chunked so a 614-residue target stays in memory."""
    best = np.inf
    for i in range(0, len(a), chunk):
        d = np.sqrt(((a[i : i + chunk, None, :] - b[None, :, :]) ** 2).sum(-1))
        best = min(best, float(d.min()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="an arm folder, or a directory of .cif complexes")
    ap.add_argument("--ca-cutoff", type=float, default=8.0,
                    help="CA-CA distance counted as backbone interface contact")
    args = ap.parse_args()

    d = args.path
    if os.path.isdir(os.path.join(d, "2_Refolded", "Complexes")):
        d = os.path.join(d, "2_Refolded", "Complexes")
    cifs = sorted(glob.glob(os.path.join(d, "*.cif")))
    if not cifs:
        sys.exit("no .cif under %s" % d)

    print("%-28s %10s %10s %9s %9s %s" % (
        "complex", "chain CA", "", "minCA", "minHeavy", "binder res with CA<%.1fA" % args.ca_cutoff))
    n_no_iface = 0
    for f in cifs:
        ch = read_chains(f)
        names = sorted(ch)
        label = os.path.basename(f)[-28:]
        if len(names) < 2:
            print("%-28s ONE CHAIN %s -- cannot be an interface" % (label, {c: len(ch[c]["ca"]) for c in names}))
            continue
        # the longer chain is the target, the shorter the binder
        names.sort(key=lambda c: -len(ch[c]["ca"]))
        tgt, bnd = ch[names[0]], ch[names[1]]
        dca = np.sqrt(((bnd["ca"][:, None, :] - tgt["ca"][None, :, :]) ** 2).sum(-1))
        contacts = int((dca < args.ca_cutoff).any(1).sum())
        n_no_iface += contacts == 0
        print("%-28s %4d/%-4d %9.2f %9.2f   %d" % (
            label, len(tgt["ca"]), len(bnd["ca"]),
            float(dca.min()), min_dist(bnd["all"], tgt["all"]), contacts))

    print("\n%d of %d complexes have NO binder residue within %.1f A (CA) of the target."
          % (n_no_iface, len(cifs), args.ca_cutoff))
    print("Chain CA counts above are the check against truncation: both chains full length")
    print("means the refold handled the complex and a zero interface is the structure, not a bug.")


if __name__ == "__main__":
    main()
