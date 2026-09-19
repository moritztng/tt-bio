"""Every parameter the device model holds must reach the optimizer, and must get a gradient.

Two defects found together on 2026-09-19, both invisible to every forward-parity gate the port
has, because both are properties of the BACKWARD and of the parameter set rather than of the
forward's output.

* `_parameters` had a `depth > 4` cut-off and the angle resnet's inner linears sit five
  containers deep, so 64 of the model's 500 parameters were never in the optimizer;
* the run's starting weights were all zero, so 356 of 436 parameters had an identically zero
  gradient for 1,388 steps.
"""

from __future__ import annotations

import pytest
import torch

import tt_bio.abodybuilder3 as abb3
from tt_bio.abodybuilder3_reference import ABB3Config
from tt_bio.autograd import Tensor
from tt_bio.train.abb3_run import _assert_every_parameter_trains
from tt_bio.train.abodybuilder3_step import _parameters


class _Uploaded:
    """A stand-in for a device tensor. The walk reads nothing but the object's identity."""

    def __init__(self, t: torch.Tensor):
        self.t = t.contiguous().float()

    @property
    def shape(self):
        return tuple(self.t.shape)


@pytest.fixture(scope="module")
def device_free_model():
    """`DeviceABB3` with the uploads faked, so the parameter set can be read without a card.

    The layout is what `_parameters` walks, and the layout is host-side: which attribute holds
    which tensor does not depend on a device existing. Requiring one would put this check
    somewhere CI cannot reach, which is where the depth cap survived for as long as it did.
    """
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]

    real = abb3.to_device_fp32
    abb3.to_device_fp32 = lambda t: _Uploaded(t)
    try:
        spec = importlib.util.spec_from_file_location(
            "_repro_for_params", root / "scripts" / "abb3_port" / "repro.py")
        repro = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(repro)
        cfg = ABB3Config(use_plddt=False)
        uploaded = []

        def to_device(t):
            p = Tensor(abb3.to_device_fp32(t), requires_grad=True)
            uploaded.append(p)
            return p

        model = abb3.DeviceABB3(repro.initial_state_dict(cfg, 0), cfg, to_device=to_device)
        return model, uploaded
    finally:
        abb3.to_device_fp32 = real


def test_the_walk_finds_every_parameter_that_was_uploaded(device_free_model):
    """Counted at the upload rather than by a second walk, so the check is independent.

    `to_device` is called exactly once per parameter, so the number of calls is the model's
    parameter count and owes nothing to the traversal under test. Under the old `depth > 4` cap
    this found 436 of 500.
    """
    model, uploaded = device_free_model
    found = _parameters(model)
    assert len(found) == len(uploaded)
    assert {id(p) for p in found} == {id(p) for p in uploaded}


def test_the_angle_resnet_blocks_are_reached(device_free_model):
    """The tensors the depth cap dropped, named, so a cap coming back fails on the right thing."""
    model, _ = device_free_model
    found = {id(p) for p in _parameters(model)}
    deep = [t for resnet in model.angles for weights, biases in resnet.blocks
            for t in list(weights) + list(biases)]
    assert deep, "the angle resnet has no inner blocks; this test is reading the wrong layout"
    assert all(id(t) in found for t in deep)


class _FakeStep:
    def __init__(self, dead, total):
        self.ungradiented = list(dead)
        self.params = list(range(total))


def test_the_gate_passes_when_every_parameter_has_a_gradient():
    _assert_every_parameter_trains(_FakeStep([], 436), gs=1, rank=0)


def test_the_gate_stops_a_run_whose_parameters_get_no_gradient():
    """The first `base-loss` leg's exact state: 356 of 436, from step 1."""
    with pytest.raises(RuntimeError, match="356 of 436 parameters received no gradient"):
        _assert_every_parameter_trains(_FakeStep(range(356), 436), gs=1, rank=0)


def test_the_gate_fires_on_a_single_dead_parameter():
    """One frozen tensor is the depth-cap defect's signature and has to fail too, not just 356."""
    with pytest.raises(RuntimeError, match="1 of 500 parameters"):
        _assert_every_parameter_trains(_FakeStep([499], 500), gs=1, rank=0)
