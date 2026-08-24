"""RF3 host featurizer parity, on committed fixtures, with no device and no foundry install.

`scripts/rf3_port/parity_gate.py` scores tt-bio's vendored RF3 featurizer against ten
captures of the upstream pipeline (one per capability class: protein monomer and multimer,
DNA, RNA, ligands in all three input forms, covalent glycan, non-canonical residues, MSA,
cyclic chains, templates). The captures are committed and 4.8 MB total, so this is the one
RF3 gate that runs anywhere -- a laptop, CI, a host with no card.

GAP_ENV is a skip, not a failure. Two key sets are only comparable against a capture made
in the same environment: `feats/ref_pos` and friends are RDKit-generated conformers, and
`coord_atom_lvl_to_be_noised` carries a random rigid augmentation built from
`torch.linalg.qr`, whose sign convention is the LAPACK backend's. The gate names every
dependency that differs and excuses only the key set that dependency owns -- a torch
difference does not excuse a conformer. Reporting an environment difference as a failure
would be the same inversion as guarding on a fixture's existence while depending on its
contents (see tests/of3_golden.py).
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "rf3_port"))
sys.path.insert(0, str(REPO / "tests"))
from _port_module import port_module  # noqa: E402


def test_rf3_featurizer_parity():
    featurizer_parity = port_module("rf3_port", "parity_gate").featurizer_parity

    rep = featurizer_parity()
    if rep["verdict"] == "GAP_ENV":
        pytest.skip("RDKit differs from the captures' ({}), and every mismatch is "
                    "RDKit-derived: {}".format(rep["env_mismatch"], rep["fixtures_pass"]))
    assert rep["verdict"] == "PASS", rep
    # the fixture count is asserted so a fixture that stops being discovered fails loudly
    # rather than passing a gate over nine of ten capability classes
    assert rep["fixtures_total"] == 10, rep["fixtures_total"]
    assert rep["fixtures_pass"] == rep["fixtures_total"], rep
    assert rep["keys_bitexact"] == rep["keys_total"], rep


def test_an_env_difference_only_excuses_its_own_keys():
    """A torch difference must not excuse a conformer, and an RDKit difference must not
    excuse a rotation. Without this the leg would go green on a real port defect the
    moment any dependency drifted."""
    g = port_module("rf3_port", "parity_gate")
    assert g.excusable_keys(None) == set()
    assert g.excusable_keys({"torch": {}}) == g.TORCH_QR_DERIVED_KEYS
    assert g.excusable_keys({"rdkit": {}}) == g.RDKIT_DERIVED_KEYS
    assert g.excusable_keys({"numpy": {}}) == set()
    assert g.excusable_keys({"torch": {}, "rdkit": {}}) == (
        g.TORCH_QR_DERIVED_KEYS | g.RDKIT_DERIVED_KEYS)
    assert not (g.TORCH_QR_DERIVED_KEYS & g.RDKIT_DERIVED_KEYS)
