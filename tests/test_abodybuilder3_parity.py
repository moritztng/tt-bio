"""ABodyBuilder3 on-device parity tests vs the reference golden.

Golden: ~/abb3_golden.pkl (or $ABB3_GOLDEN), captured by
scripts/abb3_golden.py -- the vendored reference ``StructureModule`` forward on the
paired 6yio H0-L0 Fv with the real ``plddt-loss`` checkpoint, dumping per-block
IPA / LayerNorm / Transition / BackboneUpdate / AngleResnet intermediates plus the
final single state, pLDDT logits, and atom14 positions. The ttnn port PCC-gates
each component against this golden (PCC > 0.98) -- real weights, real inputs.

This file currently ships the comparator + a reference-self-consistency test
(running the reference IPA on the golden block-0 inputs reproduces the golden
ipa_delta bit-identically, PCC ~ 1.0). That proves the golden + comparator are
correct so the ttnn IPA port (next chunk -- IPA has no reusable primitive in
tt-bio, so it is ported from scratch) drops straight in: replace the reference
call with the ttnn module and assert the same PCC bar.
"""
import os
import pickle

import pytest
import torch

_GOLD = os.environ.get("ABB3_GOLDEN", os.path.expanduser("~/abb3_golden.pkl"))
pytestmark = pytest.mark.skipif(not os.path.exists(_GOLD),
                                reason="abb3 golden pkl missing (run scripts/abb3_golden.py)")


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


@pytest.fixture(scope="module")
def golden():
    return pickle.load(open(_GOLD, "rb"))


def test_golden_shapes(golden):
    """Sanity: the golden has all 8 blocks with every component's intermediates."""
    assert len(golden["blocks"]) == 8
    b0 = golden["blocks"][0]
    assert b0["ipa_s_in"].shape == (1, 229, 128)
    assert b0["ipa_z_in"].shape == (1, 229, 229, 128)
    assert b0["ipa_rot_mats"].shape == (1, 229, 3, 3)
    assert b0["ipa_trans"].shape == (1, 229, 3)
    assert b0["ipa_delta"].shape == (1, 229, 128)
    assert b0["bb_update"].shape == (1, 229, 6)
    assert b0["ang_norm"].shape == (1, 229, 7, 2)
    assert golden["final"]["single"].shape == (1, 229, 128)
    assert golden["final"]["plddt_logits"].shape == (1, 229, 50)
    assert golden["final"]["atom14"].shape == (1, 229, 14, 3)


def test_reference_ipa_self_consistent(golden):
    """Reference IPA on the golden block-0 inputs reproduces the golden ipa_delta
    bit-identically. This validates the golden + comparator; the ttnn IPA port
    replaces the reference call below with the on-device module and asserts the
    same PCC > 0.98 bar."""
    from tt_bio._vendor.abodybuilder3.openfold.model.structure_module import StructureModule
    from tt_bio._vendor.abodybuilder3.openfold.utils.rigid_utils import Rigid, Rotation
    from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights

    cache = os.environ.get("TT_BIO_CACHE", os.path.expanduser("~/.ttbio"))
    sd = torch.load(ensure_abb3_weights(cache) if not os.path.exists(
        os.path.join(cache, "abodybuilder3_plddt.pt")) else
        os.path.join(cache, "abodybuilder3_plddt.pt"),
        map_location="cpu", weights_only=True)
    model = StructureModule(**ABB3_CONFIG)
    model.load_state_dict(sd, strict=True)
    model.eval()

    b0 = golden["blocks"][0]
    r = Rigid(Rotation(rot_mats=b0["ipa_rot_mats"], quats=None), b0["ipa_trans"])
    with torch.no_grad():
        delta = model.ipa_layers[0](
            b0["ipa_s_in"], b0["ipa_z_in"], r, b0["ipa_mask"])
    pcc = _pcc(delta.float(), b0["ipa_delta"].float())
    assert pcc > 0.999, f"reference IPA self-consistency PCC={pcc}"


@pytest.mark.skip(reason="ttnn IPA port is the next chunk -- IPA has no reusable "
                         "primitive in tt-bio (its ESMFold2 is a diffusion folder), "
                         "so InvariantPointAttention is ported from scratch.")
def test_abodybuilder3_ipa_on_device(golden):
    """On-device IPA PCC > 0.98 vs the golden ipa_delta, fed the golden block-0
    (s, z, rot_mats, trans, mask). Drop-in target once tt_bio.abodybuilder3.IPA
    lands."""
