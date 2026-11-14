"""RFD4's 147-class atom37 vocabulary is checkpoint-load-bearing: pin it to upstream's bytes and shape."""

import hashlib
from importlib.resources import files

from tt_bio._vendor.rfd4_prep._cif.encoding import RESIDUES_BY_TOKEN_TYPE, UNIFIED_ATOM37_ENCODING
from tt_bio._vendor.rfd4_prep._cif.utils_constants import _build_unified_atom37_mask
from tt_bio.rfd4.ae_fold import N_RES

# sha256 of src/rfproteina/resources/aa_rna_dna_glycan_ncaa.yaml in RFD4-Proteina 5248e9a1 (unchanged at 2db07bb4)
UPSTREAM_SHA256 = "0efd6c84938c2d944f24587769886d4cbd0e0a248106274bff12c64dede6d3a1"


def test_vocabulary_is_upstreams_bytes():
    data = files("tt_bio._vendor.rfd4_prep").joinpath("resources/aa_rna_dna_glycan_ncaa.yaml").read_bytes()
    assert hashlib.sha256(data).hexdigest() == UPSTREAM_SHA256


def test_147_classes_in_checkpoint_order():
    counts = {group: len(names) for group, names in RESIDUES_BY_TOKEN_TYPE.items()}
    assert counts == {"mask": 1, "aa": 21, "rna": 5, "dna": 5, "atomized": 1, "unknown": 1, "ncaa": 100, "glycan": 13}
    tokens = list(UNIFIED_ATOM37_ENCODING.tokens)
    assert len(tokens) == N_RES == 147
    assert tokens[:3] == ["<M>", "ALA", "ARG"] and tokens[21:24] == ["UNK", "A", "C"]


def test_atom37_mask_matches_layouts():
    mask = _build_unified_atom37_mask()
    assert mask.shape == (147, 37)
    for i, name in enumerate(UNIFIED_ATOM37_ENCODING.tokens):
        slots = UNIFIED_ATOM37_ENCODING.token_atoms[name]
        assert len(slots) == 37
        assert [bool(a) for a in slots] == [bool(v) for v in mask[i]], name
    assert mask[0].sum() == 0  # the mask token has no atoms
