#!/usr/bin/env python3
"""How the frame mask's cost scales with atom count.

`get_token_frame_atoms` builds an N_atom x N_atom distance matrix and a topk over it, per
diffusion sample, on the host. At the 832-864 atoms the two openbind co-folds carry it is
0.37-0.46 % of the fold, which is nothing; the question `land-standing` needs answered is
what it is on a target ten times that size, and a quadratic term answers that badly.

Host-only and therefore load-sensitive: the host was NOT quiet when this ran, so the absolute
milliseconds are an upper bound. The EXPONENT is what this is for, and a fitted exponent is
far more robust to load than any single time is.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("WT", "/home/ttuser/.coworker/wt/of3t-d10-d107"))
from tt_bio._vendor.openfold3.core.utils.atomize_utils import (      # noqa: E402
    get_token_frame_atoms)

ATOMS_PER_RES = 4
SIZES = (512, 1024, 2048, 4096, 8192)
REPEATS = 3


def build(n_atom):
    n_res = n_atom // ATOMS_PER_RES
    n_tok = n_res
    napt = torch.full((n_tok,), float(ATOMS_PER_RES))
    start = (torch.arange(n_res) * ATOMS_PER_RES).float()
    batch = {
        "token_mask": torch.ones(n_tok),
        "num_atoms_per_token": napt,
        "asym_id": torch.zeros(n_tok),
        "start_atom_index": start,
        "is_protein": torch.ones(n_tok),
        "is_dna": torch.zeros(n_tok),
        "is_rna": torch.zeros(n_tok),
        "is_atomized": torch.zeros(n_tok),
        "restype": torch.nn.functional.one_hot(
            torch.zeros(n_tok, dtype=torch.long), 32).float(),
        "atom_mask": torch.ones(n_res * ATOMS_PER_RES),
    }
    g = torch.Generator().manual_seed(107)
    x = torch.randn(n_res * ATOMS_PER_RES, 3, generator=g) * 20.0
    return batch, x


if __name__ == "__main__":
    rows = []
    for n in SIZES:
        batch, x = build(n)
        best = None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            get_token_frame_atoms(batch=batch, x=x, atom_mask=batch["atom_mask"])
            ms = (time.perf_counter() - t0) * 1e3
            best = ms if best is None else min(best, ms)
        rows.append({"n_atom": int(x.shape[0]), "ms_best_of_%d" % REPEATS: round(best, 2)})
        print(json.dumps(rows[-1]), flush=True)
    # log-log slope over the whole ladder, and over the top two rungs where the quadratic
    # term should dominate if there is one
    import math
    def slope(a, b):
        return ((math.log(b["ms_best_of_%d" % REPEATS]) - math.log(a["ms_best_of_%d" % REPEATS]))
                / (math.log(b["n_atom"]) - math.log(a["n_atom"])))
    out = {"rows": rows, "exponent_full_ladder": round(slope(rows[0], rows[-1]), 3),
           "exponent_top_two_rungs": round(slope(rows[-2], rows[-1]), 3),
           "host_quiet": os.environ.get("HOST_QUIET", "NO -- absolute ms are an upper bound")}
    print(json.dumps(out, indent=1))
    with open(os.environ.get("OUT", "/tmp/FRAME_COST.json"), "w") as fh:
        json.dump(out, fh, indent=1)
