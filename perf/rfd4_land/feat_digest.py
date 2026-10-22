"""Digest RFD4's host featurization of one spec, key by key, card-free.

    PYTHONPATH=<tree> python perf/rfd4_land/feat_digest.py SPEC OUTDIR > digest.json

Two trees (or two venvs) that featurize the same spec to the same digests feed the trunk the
same input, so any design difference between them is downstream of featurization.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from tt_bio import rfd4

spec, out = sys.argv[1], Path(sys.argv[2])
designs = rfd4.prepare_examples(spec, out)
features, atom_array = rfd4.featurize(designs[0].cif, seed=0)


def digest(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return f"{v.dtype}{list(v.shape)}:{hashlib.md5(np.ascontiguousarray(v).tobytes()).hexdigest()[:12]}"
    return repr(v)[:80]


coords = np.asarray(atom_array.coord)
rec = {k: digest(v) for k, v in sorted(features.items())}
rec["_atom_array"] = f"{len(atom_array)} atoms, coord md5 {hashlib.md5(np.nan_to_num(coords).tobytes()).hexdigest()[:12]}"
rec["_torch"] = torch.__version__
print(json.dumps(rec, indent=1))
