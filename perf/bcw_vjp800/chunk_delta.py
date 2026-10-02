#!/usr/bin/env python3
"""What the chunking in `lowmem.py` does to the reading, at the size the grade is read at.

Takes two `ref` caches built with the same (seed, n, blocks) and different `--floor`, and
prints the float64 gradients' agreement block by block plus the move in each torch arm's
rel L2. The claim this is evidence for: the chunked reference is the same reference.
"""
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad.afgrad import rel_l2        # noqa: E402

a, b = (pathlib.Path(p) for p in sys.argv[1:3])
ma, mb = (json.loads((p / "meta.json").read_text()) for p in (a, b))
assert (ma["n"], ma["seed"], ma["blocks"]) == (mb["n"], mb["seed"], mb["blocks"]), "not a pair"
print(f"== chunk_delta n={ma['n']} floor {ma['floor']} vs {mb['floor']}")
for j in ma["blocks"]:
    xa = torch.load(a / f"b{j}.pt", weights_only=False)
    xb = torch.load(b / f"b{j}.pt", weights_only=False)
    tag = xa["block"]
    for k in ("dz_ref", "dm_ref", "fwd_z", "fwd_m"):
        if k in xa:
            print(f"{tag:>6} {k:>7}  rel_l2 {rel_l2(xa[k], xb[k]):.3e}  "
                  f"exact {bool(torch.equal(xa[k], xb[k]))}")
    for k, v in ma["arms"][tag].items():
        print(f"{tag:>6} {k:>16}  {v['rel_l2']:.6f} vs {mb['arms'][tag][k]['rel_l2']:.6f}")
