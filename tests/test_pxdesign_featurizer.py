"""The PXDesign design featurizer must match upstream bit-exactly.

`conditional_templ` is where this port breaks quietly: the model is conditioned on a 64-bin
distogram of the target, written only into the sub-block of resolved non-`xpb` tokens, and if
the binder placeholder leaks into that block the model is handed the answer and still returns
a plausible structure. Nothing downstream complains. So it is gated here, against a committed
capture of the upstream featurizer on the PD-L1 quick-start target.

Device-free and install-free: the reference is committed
(`scripts/pxdesign_port/parity_artifacts/pdl1/`), so this needs no upstream PXDesign, no
protenix and no card.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from tt_bio.pxdesign.featurize import (RESTYPE_VOCAB, condition_template,
                                       condition_template_index, restype_onehot)

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "pxdesign_port" / "parity_gate.py"
ART = REPO / "scripts" / "pxdesign_port" / "parity_artifacts" / "pdl1"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("pxdesign_parity_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def report(gate):
    if not (ART / "ref_design_f.pt").exists():
        pytest.skip("committed PD-L1 capture missing")
    return gate.featurizer_parity()


def test_gate_passes(report):
    assert report["verdict"] == "PASS", report["mismatches"]
    assert report["checks_passed"] == report["checks_total"] > 0


def test_fixture_is_the_pdl1_anchor(report):
    """196 tokens = 116 cropped target + an 80-residue binder, the paper's own PD-L1 cell."""
    assert report["n_token"] == 196
    assert report["n_binder_placeholder_tokens"] == 80
    assert report["n_conditioned_tokens"] == 116


def test_gate_fails_when_the_binder_placeholder_leaks(gate, monkeypatch):
    """The arm must fail on the bug it is named after, or it is decoration."""
    import tt_bio.pxdesign.featurize as F
    monkeypatch.setattr(F, "BINDER_PLACEHOLDER", "___not_a_residue___")
    assert gate.featurizer_parity()["verdict"] == "FAIL"


def test_gate_fails_on_the_wrong_bin_edge_count(gate, monkeypatch):
    """64 bins means 63 interior boundaries. Off by one and every bin index shifts."""
    import tt_bio.pxdesign.featurize as F
    monkeypatch.setattr(F, "N_TEMPL_BINS", 65)
    assert gate.featurizer_parity()["verdict"] == "FAIL"


def test_lookup_index_reserves_row_zero():
    """The embedding has 65 rows for 64 bins because 0 means 'no condition here'."""
    templ = torch.tensor([[0, 63], [63, 0]])
    mask = torch.tensor([[False, True], [True, False]])
    idx = condition_template_index(templ, mask)
    assert idx.tolist() == [[0, 64], [64, 0]]


def test_restype_vocabulary_is_36_way_with_the_design_tokens_last():
    assert len(RESTYPE_VOCAB) == 36
    assert RESTYPE_VOCAB[32:] == ("xpb", "xpa", "rbb", "raa")
    oh = restype_onehot(["ALA", "xpb"])
    assert oh.shape == (2, 36) and oh[0, 0] == 1.0 and oh[1, 32] == 1.0


@pytest.mark.parametrize("bad,msg", [
    (dict(coord=torch.zeros(0, 3), res_name=[], mol_type=[], is_resolved=torch.zeros(0)),
     "zero tokens"),
    (dict(coord=torch.zeros(3, 3), res_name=["ALA"], mol_type=["protein"] * 3,
          is_resolved=torch.ones(3)), "res_names"),
    (dict(coord=torch.zeros(3, 4), res_name=["ALA"] * 3, mol_type=["protein"] * 3,
          is_resolved=torch.ones(3)), "N_token, 3"),
])
def test_malformed_input_raises_rather_than_designing_against_nothing(bad, msg):
    """tt-bio-fold-succeeds-on-malformed-input: an empty target must not fold as garbage."""
    with pytest.raises(ValueError, match=msg):
        condition_template(**bad)


def test_unknown_residue_is_refused_not_silently_dropped():
    with pytest.raises(ValueError, match="outside the 36-way"):
        restype_onehot(["ALA", "NOT_A_RESIDUE"])
