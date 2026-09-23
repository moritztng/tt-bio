"""CPU regression: OpenDDE nucleotides split into backbone/base structural tokens.

Upstream AtomArrayTokenizer.tokenize_structural gives a standard DNA/RNA residue a backbone
token (NUCLEIC_BACKBONE_ATOMS) and a base token, twinned; the unknown nucleotide, which has
no base atoms, stays one backbone token. Full index parity against upstream's own featurizer
is scripts/opendde_structtoken_featurizer_parity.py, which needs an opendde venv.
"""
import os

import pytest

from tt_bio.opendde_data import (NUCLEIC_BACKBONE_ATOMS, STRUCTURAL_TOKEN_ROLES,
                                 _atom_names, build_structural_token_features)
from tt_bio.protenix_data import build_complex_features

_MOL_DIR = os.path.expanduser("~/.boltz/mols")
pytestmark = pytest.mark.skipif(
    not os.path.exists(_MOL_DIR), reason="needs bundled CCD mol library (~/.boltz/mols)")


@pytest.mark.parametrize("mt,seq", [("rna", "GACUN"), ("dna", "GACTN")])
def test_nucleotide_backbone_base_split(mt, seq):
    feats = build_complex_features([("MKGW", None, "protein"), (seq, None, mt)], mol_dir=_MOL_DIR)
    ifd = build_structural_token_features(feats)
    role, parent, twin = ifd["subtoken_role_id"], ifd["parent_residue_idx"], ifd["twin_token_idx"]
    bb, base = STRUCTURAL_TOKEN_ROLES[f"{mt}_bb"], STRUCTURAL_TOKEN_ROLES[f"{mt}_base"]
    na = [i for i, p in enumerate(parent.tolist()) if p >= 4]
    # four standard nucleotides split in two, the unknown one (N / DN) stays whole
    assert role[na].tolist() == [bb, base] * 4 + [bb]
    for i in na[:-1]:
        assert int(twin[twin[i]]) == i and int(parent[twin[i]]) == int(parent[i])
    assert int(twin[na[-1]]) == -1
    # every atom lands in the token its name says, numbered from 0 within that token
    a2s, a2sa = ifd["atom_to_structural_token_idx"], ifd["atom_to_structural_tokatom_idx"]
    names = _atom_names(feats)
    for a, t in enumerate(a2s.tolist()):
        if int(role[t]) in (bb, base):
            assert (names[a] in NUCLEIC_BACKBONE_ATOMS) == (int(role[t]) == bb or t == na[-1])
    for t in range(len(parent)):
        slots = sorted(a2sa[a2s == t].tolist())
        assert slots == list(range(len(slots)))
    # the protein chain's split is unchanged: M K W split, G stays whole
    assert role[[i for i, p in enumerate(parent.tolist()) if p < 4]].tolist() == [1, 2, 1, 2, 1, 1, 2]
