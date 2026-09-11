"""How many real heavy atoms does each cdk2x2_N fixture have, and what does the fold pad to?

CPU only, no device. Answers P5's first two questions: the real atom count from the production
featuriser, and the bucket the shipped diffusion path actually allocates
(``_populate_diffusion_cache``: ``padded_seq * MAX_ATOMS_PER_TOKEN``, tenstorrent.py:9548).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from tt_bio import main as M
from tt_bio.data import const
from tt_bio.data.featurizer import Boltz2Featurizer
from tt_bio.data.mol import load_canonicals
from tt_bio.data.tokenize import Boltz2Tokenizer
from tt_bio.tenstorrent import (ATOM_WINDOW, MAX_ATOMS_PER_TOKEN,
                                PAIRFORMER_PAD_MULTIPLE)
from tt_bio.token_axis import pad_amount

FIX = Path(__file__).resolve().parents[1] / "size512" / "fixtures"


def census(name, ccd, mol_dir, msa_dir, tok, feat):
    feats, _ = M.prepare_features(
        FIX / f"{name}.yaml", ccd, mol_dir, msa_dir, tok, feat,
        use_msa=False, msa_url=None, msa_strategy="greedy", msa_user=None,
        msa_pass=None, api_key=None, max_msa=None, single_sequence=True,
    )
    n_atom = int(feats["atom_pad_mask"].sum().item())
    n_atom_feat = int(feats["atom_pad_mask"].shape[-1])
    n_token = int(feats["token_pad_mask"].shape[-1])
    token_pad = pad_amount(n_token, PAIRFORMER_PAD_MULTIPLE)
    padded_seq = n_token + token_pad
    n_padded = padded_seq * MAX_ATOMS_PER_TOKEN
    if n_atom_feat > n_padded:                      # the nucleic-acid escape branch
        n_padded = -(-n_atom_feat // ATOM_WINDOW) * ATOM_WINDOW
    smallest = -(-n_atom // ATOM_WINDOW) * ATOM_WINDOW
    return {
        "fixture": name,
        "n_token": n_token,
        "padded_seq": padded_seq,
        "n_atom_real": n_atom,               # atoms the featuriser emitted, unpadded
        "n_atom_featurised": n_atom_feat,    # after the featuriser's own ceil-to-32
        "n_atom_device": n_padded,           # what the diffusion path allocates
        "atoms_per_token_real": round(n_atom / n_token, 3),
        "smallest_legal_32": smallest,
        "pad_factor_device": round(n_padded / n_atom, 4),
        "windows_device": n_padded // ATOM_WINDOW,
        "windows_smallest": smallest // ATOM_WINDOW,
    }


def main(names):
    mol_dir = Path.home() / ".boltz" / "mols"
    ccd = load_canonicals(mol_dir)
    msa_dir = Path("/tmp/b2x_atom_census_msa")
    msa_dir.mkdir(parents=True, exist_ok=True)
    tok, feat = Boltz2Tokenizer(), Boltz2Featurizer()
    rows = [census(n, ccd, mol_dir, msa_dir, tok, feat) for n in names]
    print(json.dumps(rows, indent=2))
    out = Path(__file__).with_name("atom_census.json")
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:] or ["cdk2x2_298", "cdk2x2_512"])
