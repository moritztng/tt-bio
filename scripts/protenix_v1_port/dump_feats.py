#!/usr/bin/env python3
"""Featurize a target with tt-bio's own featurizer and write the feature dict to disk.

Both parity arms read this file, so the module-level PCCs measure the PORT and not the
featurizer. The featurizer itself is scored separately (feat_parity.py) against upstream's
v0.5.0 data pipeline.

    PYTHONPATH=$WT env/bin/python3 scripts/protenix_v1_port/dump_feats.py \
        examples/multimer.yaml /tmp/pv1/feats.pt [msa.a3m]

A third argument attaches that alignment to every PROTEIN chain, which is what the
release-gate reference fold needs: its device arm folds MSA-on, so a single-sequence
reference would be scoring two different workloads. Without it the capture stays
single-sequence and byte-reproducible, which is what the module-PCC arm wants.
"""
import sys
from pathlib import Path

import torch

from tt_bio.main import _read_bio_chains, _read_bio_constraints
from tt_bio.protenix_data import build_complex_features
from tt_bio import weights


def main(src, out, a3m=None):
    chains = _read_bio_chains(Path(src))
    bonds = _read_bio_constraints(Path(src))
    # Single-sequence unless an a3m is given: no MSA search either way, so the capture stays
    # reproducible byte for byte. build_complex_features takes the alignment TEXT, and only
    # protein chains carry one (the featurizer refuses an MSA on a nucleic-acid chain).
    text = Path(a3m).read_text() if a3m else None
    specs = [(seq, text if mt == "protein" else None, mt) for _cid, seq, _spec, mt in chains]
    ids = [cid for cid, _s, _sp, _mt in chains]
    feats = build_complex_features(specs, mol_dir=str(weights.fetch("mols")),
                                   chain_ids=ids, bonds=bonds)
    torch.save({"feats": feats, "chains": chains}, out)
    print("N_token", feats["residue_index"].shape[-1], "N_atom", feats["ref_pos"].shape[0])
    print("keys", sorted(feats))
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
