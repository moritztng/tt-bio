#!/usr/bin/env python3
"""Featurize an input through tt-bio's Protenix path, ring closure included, for
protenix_v1_upstream_ring.sh. Records the N of the first residue and the C of the last."""
import sys, torch
from pathlib import Path
from tt_bio.main import _read_bio_chains, _read_bio_bonds
from tt_bio.protenix_data import build_complex_features
from tt_bio import weights
src, out = Path(sys.argv[1]), sys.argv[2]
chains = _read_bio_chains(src)
bonds = _read_bio_bonds(src, chains)
print("bonds", bonds)
specs = [(seq, None, mt) for _c, seq, _s, mt, _m in chains]
f = build_complex_features(specs, mol_dir=str(weights.fetch("mols")),
                           chain_ids=[c[0] for c in chains], bonds=bonds)
ch = f["ref_atom_name_chars"].reshape(f["ref_pos"].shape[0], 4, 64).argmax(-1) + 32
names = ["".join(chr(c) for c in r).strip() for r in ch.tolist()]
tok = f["atom_to_token_idx"].tolist()
i_n = next(i for i, (n, t) in enumerate(zip(names, tok)) if t == 0 and n == "N")
i_c = next(i for i, (n, t) in enumerate(zip(names, tok)) if t == max(tok) and n == "C")
print("N_atom", len(names), "i_n", i_n, "i_c", i_c, "OXT" in names, "token_bonds", int(f["token_bonds"].sum()))
torch.save({"feats": f, "chains": chains, "i_n": i_n, "i_c": i_c}, out)
