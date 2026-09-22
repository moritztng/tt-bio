#!/usr/bin/env python3
"""Does the mask actually build, with the dtypes the shipped feature dict carries?

`get_token_frame_atoms` indexes, gathers and subtracts masks, and a bool where it wants a
float raises rather than returning something wrong. This drives it through the exact call
`openfold3_fold._confidence` now makes, on a small batch shaped like the real feature dict:
standard protein residues plus an atomized ligand, so both branches of the mask run.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.environ.get("WT", "/home/ttuser/.coworker/wt/of3t-d10-d107"))
from tt_bio._vendor.openfold3.core.utils.atomize_utils import (      # noqa: E402
    get_token_frame_atoms)

N_RES, N_LIG = 12, 4                       # 12 standard residues + 4 atomized ligand atoms
ATOMS_PER_RES = 4                          # N, CA, C, O in the token-atom layout


def build():
    n_tok = N_RES + N_LIG
    napt = torch.tensor([ATOMS_PER_RES] * N_RES + [1] * N_LIG, dtype=torch.float32)
    start = torch.cat([torch.arange(N_RES) * ATOMS_PER_RES,
                       N_RES * ATOMS_PER_RES + torch.arange(N_LIG)]).float()
    n_atom = int(napt.sum())
    batch = {
        "token_mask": torch.ones(n_tok),
        "num_atoms_per_token": napt,
        "asym_id": torch.cat([torch.zeros(N_RES), torch.ones(N_LIG)]),
        "start_atom_index": start,
        "is_protein": torch.cat([torch.ones(N_RES), torch.zeros(N_LIG)]),
        "is_dna": torch.zeros(n_tok),
        "is_rna": torch.zeros(n_tok),
        "is_atomized": torch.cat([torch.zeros(N_RES), torch.ones(N_LIG)]),
        # One-hot over the 32 restypes, as `get_token_atom_index_offset` requires. Index
        # 0 is ALA, whose N/CA/C offsets are the ones the protein branch reads.
        "restype": torch.nn.functional.one_hot(
            torch.zeros(n_tok, dtype=torch.long), 32).float(),
        "atom_mask": torch.ones(n_atom),
    }
    g = torch.Generator().manual_seed(24)
    x = torch.randn(n_atom, 3, generator=g) * 4.0
    return batch, x


def collinear(x):
    """The negative control. A mask that is all-ones on every input it ever sees is not a
    mask, so put the four ligand atoms on a line: the a-b-c angle then falls outside
    [25, 155] degrees and upstream's angle constraint has to reject them. If this arm also
    reads 0 rejected, the mask is not live and the fix is decoration."""
    x = x.clone()
    base = N_RES * ATOMS_PER_RES
    for i in range(N_LIG):
        x[base + i] = torch.tensor([float(i) * 1.4, 0.0, 0.0])
    return x


def run(batch, x, label):
    _, mask = get_token_frame_atoms(batch=batch, x=x,
                                    atom_mask=batch["atom_mask"].float())
    mask = mask.bool()
    return {"arm": label, "tokens": int(mask.numel()),
            "standard_residues": N_RES, "atomized_tokens": N_LIG,
            "frames_valid": int(mask.sum()),
            "frames_rejected": int((~mask).sum()),
            "rejected_are_all_atomized": bool((~mask)[:N_RES].sum() == 0),
            "mask": mask.int().tolist()}


if __name__ == "__main__":
    batch, x = build()
    out = [run(batch, x, "random ligand geometry"),
           run(batch, collinear(x), "collinear ligand (negative control)")]
    assert out[1]["frames_rejected"] > out[0]["frames_rejected"], (
        "the negative control did not move the mask")
    print(json.dumps({"builds": True, "arms": out}, indent=1))
