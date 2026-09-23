#!/usr/bin/env python3
"""Count distinct outputs across the repeats in a run.sh output dir and name where they part.

diff.py <dir> [<dir> ...]   (several dirs pool their folds, e.g. an inproc and a fresh run)

Prints, per fold: its pid:fold id, the sampled-coordinate hash and the CIF's CA digest; then the
number of distinct CA digests, the pairwise CA-RMSD range between distinct structures (no
superposition: every fold starts from the same noise, so a rigid-body offset is itself a
difference), and the first hashed key, in call order, whose value is not the same in every fold.
"""
import hashlib
import itertools
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np


def ca(cif):
    rows = [l.split() for l in open(cif) if l.startswith(("ATOM", "HETATM"))]
    hdr = [l.strip() for l in open(cif) if l.startswith("_atom_site.")]
    ix = {h.split(".", 1)[1]: i for i, h in enumerate(hdr)}
    xyz = [[float(r[ix[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")]
           for r in rows if r[ix["label_atom_id"]] == "CA"]
    return np.asarray(xyz)


folds, order = OrderedDict(), []
cifs = []
for d in map(Path, sys.argv[1:]):
    for line in open(d / "hash.txt"):
        head, *kv = line.split()
        fid, key = head.split("/", 1)
        fid = f"{d.name}:{fid}"
        folds.setdefault(fid, {})
        for x in kv:
            k, v = x.split("=", 1)
            folds[fid][f"{key} {k}"] = v
            if fid == next(iter(folds)):
                order.append(f"{key} {k}")
    cifs += sorted(d.glob("out/boltz2_results_*/structures/*.cif"))

digs = {}
for c in cifs:
    x = ca(c)
    digs[c] = (hashlib.sha1(np.round(x, 3).tobytes()).hexdigest()[:8], x)
print(f"folds hashed {len(folds)}, CIFs {len(cifs)}")
for c, (h, _) in digs.items():
    print(f"  {c.parent.parent.parent.parent.name}/{c.stem}: CA {h}")
groups = defaultdict(list)
for c, (h, x) in digs.items():
    groups[h].append(c)
print(f"distinct CA digests: {len(groups)}  " +
      " ".join(f"{h}x{len(v)}" for h, v in sorted(groups.items(), key=lambda t: -len(t[1]))))
reps = [digs[v[0]][1] for v in groups.values()]
if len(reps) > 1:
    r = [float(np.sqrt(((a - b) ** 2).sum(1).mean())) for a, b in itertools.combinations(reps, 2)]
    print(f"pairwise CA-RMSD between distinct structures: {min(r):.3f}-{max(r):.3f} A over {len(reps[0])} CA")

first = None
for k in order:
    vals = {f.get(k) for f in folds.values()}
    if len(vals) > 1:
        first = k
        split = defaultdict(list)
        for fid, f in folds.items():
            split[f.get(k)].append(fid)
        print(f"FIRST DIFFERENCE: {k}")
        for v, ids in split.items():
            print(f"  {v}: {len(ids)} folds  {' '.join(ids[:6])}")
        break
if first is None:
    print(f"no hashed key differs across {len(folds)} folds ({len(order)} keys)")
