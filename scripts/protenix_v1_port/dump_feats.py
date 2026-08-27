#!/usr/bin/env python3
"""Featurize a target with tt-bio's own featurizer and write the feature dict to disk.

Both parity arms read this file, so the module-level PCCs measure the PORT and not the
featurizer. The featurizer itself is scored separately (feat_parity.py) against upstream's
v0.5.0 data pipeline.

    PYTHONPATH=$WT env/bin/python3 scripts/protenix_v1_port/dump_feats.py \
        examples/multimer.yaml /tmp/pv1/feats.pt
"""
import sys
from pathlib import Path

import torch

from tt_bio.main import _read_bio_chains, _read_bio_constraints
from tt_bio.protenix_data import build_complex_features
from tt_bio import weights


def main(src, out, msa_dir=None):
    chains = _read_bio_chains(Path(src))
    bonds = _read_bio_constraints(Path(src))
    # single-sequence on purpose: no MSA search, so the capture is reproducible byte for byte
    specs = [(seq, None, mt) for _cid, seq, _spec, mt in chains]
    ids = [cid for cid, _s, _sp, _mt in chains]
    feats = build_complex_features(specs, mol_dir=str(weights.fetch("mols")),
                                   chain_ids=ids, bonds=bonds)
    torch.save({"feats": feats, "chains": chains}, out)
    print("N_token", feats["residue_index"].shape[-1], "N_atom", feats["ref_pos"].shape[0])
    print("keys", sorted(feats))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
