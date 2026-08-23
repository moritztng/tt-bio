"""Which triangle attentions take the fused SDPA, and who is allowed to decide.

`TT_BIO_TRIATT_FUSED_HIFI` is process-wide, and PXDesign runs AF2-IG and the Protenix filter in
one process, so the variable cannot be AF2's shipping switch: flipping AF2's triangle attention
with it flips Protenix's too, and protenix-v2/opendde carry their own open recommendation on a
neighbouring softmax flip. `AF2DeviceModel.triatt_fused` scopes it to AF2's own pair stacks.

Two properties hold the design up and neither needs a card:

* an attention that pinned nothing follows the variable AT CALL TIME, so the perf branch's A/B
  legs -- which assign `tenstorrent._TRIATT_FUSED_HIFI` after the model is built -- still switch
  arms rather than silently running one arm twice;
* an attention that pinned a bool ignores the variable, which is what keeps AF2's choice off
  Protenix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tt_bio import tenstorrent as TT  # noqa: E402
from tt_bio.af2 import TRIATT_FUSED_STACKS, AF2DeviceModel  # noqa: E402


@pytest.fixture
def process_default(monkeypatch):
    def set(value: bool):
        monkeypatch.setattr(TT, "_TRIATT_FUSED_HIFI", value)
    return set


@pytest.mark.parametrize("default", [False, True])
def test_unpinned_follows_the_process_default(process_default, default):
    process_default(default)
    assert TT._fused_hifi_on(None) is default


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
def test_pinned_ignores_the_process_default(process_default, default, pinned):
    process_default(default)
    assert TT._fused_hifi_on(pinned) is pinned


def test_default_model_pins_nothing():
    """The class default has to leave every stack on the variable, or it changes today's folds."""
    model = AF2DeviceModel.__new__(AF2DeviceModel)
    assert model.triatt_fused is None
    assert [model._fused_hifi(s) for s in TRIATT_FUSED_STACKS] == [None] * 3


@pytest.mark.parametrize("stacks, want", [
    (("extra_msa", "evoformer", "template"), [True, True, True]),
    (("extra_msa", "evoformer"), [True, True, False]),          # the trunk-only fallback
    ((), [False, False, False]),                                # the incumbent, pinned
])
def test_a_set_pins_exactly_those_stacks(stacks, want):
    model = AF2DeviceModel.__new__(AF2DeviceModel)
    model.device_extra_msa = model.device_evoformer = model.device_template = []
    model.set_triatt_fused(stacks)
    assert [model._fused_hifi(s) for s in TRIATT_FUSED_STACKS] == want


def test_an_unknown_stack_is_refused():
    """A typo that silently pins nothing would read as a clean A/B with both arms identical."""
    model = AF2DeviceModel.__new__(AF2DeviceModel)
    model.device_extra_msa = model.device_evoformer = model.device_template = []
    with pytest.raises(AssertionError, match="unknown pair stacks"):
        model.set_triatt_fused(("evoformer", "trunk"))
