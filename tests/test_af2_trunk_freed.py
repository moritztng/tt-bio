"""A dropped `AF2DeviceModel` is freed by its last reference, not by the cyclic collector.

BindCraft 2 swaps trunks in and out of a resident pool about a hundred times a trajectory. When
the device model sat in a reference loop (its template stack held the model's own bound `_up` and
`_down`), each evicted trunk kept its device weights until CPython's gen-2 collector happened to
run, so held DRAM at a trajectory boundary stepped up by whole trunks (issue #18, round 2). This
pins the property, not the one loop: build the device model with the card-side pieces stubbed,
drop it with the collector off, and it must already be gone.
"""
from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import tt_bio.af2 as af2  # noqa: E402


class _Block:
    """Stands in for a ttnn block; holds nothing on a card."""

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def no_card(monkeypatch):
    for name in ("AF2PairBlock", "AF2EvoformerBlock", "AF2SingleActivations"):
        monkeypatch.setattr(af2, name, _Block)
    monkeypatch.setattr(af2, "get_device", lambda: object())
    monkeypatch.setattr(af2, "compute_kernel_config", lambda: None)


@pytest.mark.parametrize("substitute", [False, True])
def test_a_dropped_device_model_needs_no_collector(no_card, substitute):
    model = af2.AF2DeviceModel(template=True, multimer=False, structure=False,
                               num_evoformer_blocks=1, num_extra_msa_blocks=1).to_device()
    if substitute:
        model._install_substitution(torch.ones(1, 8), torch.ones(8, 8))
    gone = weakref.ref(model)
    gc.collect()
    gc.disable()
    try:
        del model
        assert gone() is None, "the device model is in a reference loop"
    finally:
        gc.enable()
