"""The optimizer/tape contract: a stepped leaf is still a leaf the tape can resolve.

`autograd._PARAMS` maps the identity of the RAW handle the MODEL holds to the leaf that
owns it, and three callers in this tree legitimately replace a leaf's handle:
`AdamW.step`, `checkpoint.load_adapter` and `autograd.Tensor.free`'s L1 eviction. If the
registry does not follow, `parameter_for` returns None at every call site from that moment
on and the next backward reaches zero parameters -- with real weights, a real optimizer and
a loss curve that still falls. `of3t-modeltraj` measured that shape on the shipped path:
`grad_norm` exactly 0.0 from step 2, and a 20-step weight trajectory bit-identical to a
model that computes nothing.

No card. The device handles are stand-ins and the transfers are numpy, which is deliberate:
this is a contract between the optimizer and the tape, it is the cheapest regression signal
in the campaign, and a test that needs a card is a test a gate skips.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ttnn = pytest.importorskip("ttnn")      # imported by tt_bio.autograd; no device is opened

from tt_bio import autograd as ag                                   # noqa: E402
from tt_bio.train import optim as optim_mod                         # noqa: E402
from tt_bio.train.lora import Parameters                            # noqa: E402


class _Handle:
    """What a ttnn device tensor answers to, for the three things the optimizer asks."""

    def __init__(self, arr):
        self.arr = np.ascontiguousarray(arr, dtype=np.float32)

    def device(self):
        return "fake-device"

    @property
    def dtype(self):
        return "fake-bf16"

    @property
    def shape(self):
        return self.arr.shape


class _Model:
    """A model holding its weights where a walk finds them: on attributes."""

    def __init__(self, **weights):
        for name, h in weights.items():
            setattr(self, name, h)


@pytest.fixture(autouse=True)
def _host_transfers(monkeypatch):
    """Host<->'device' as numpy. `to_device` mints a NEW handle, which is the whole point:
    the defect is that the registry stays keyed on the one it replaced."""
    monkeypatch.setattr(optim_mod, "to_host",
                        lambda t, dtype=None: (t.arr if isinstance(t, _Handle)
                                               else np.asarray(t)).astype(
                                                   np.float32 if dtype is None else dtype))
    monkeypatch.setattr(optim_mod, "to_device",
                        lambda arr, device, dtype=None, layout=None: _Handle(arr))
    yield
    ag.forget_parameters()


def _walked(n_tensors=3, dim=4):
    """The shipped discovery's output shape: leaves plus where the walk found them."""
    rng = np.random.default_rng(20260921)
    raw = {f"w{i}": _Handle(rng.standard_normal((dim, dim))) for i in range(n_tensors)}
    model = _Model(**raw)
    params = Parameters(
        {name: ag.parameter(getattr(model, name)) for name in raw},
        slots={name: (model, name) for name in raw})
    return model, params


def _resolved(params):
    """How many walked slots the tape can still hand a gradient to."""
    out = 0
    for name, (owner, key) in params.slots.items():
        handle = owner[key] if isinstance(owner, (dict, list)) else getattr(owner, key)
        if ag.parameter_for(handle) is not None:
            out += 1
    return out


def test_walked_weights_resolve_before_any_step():
    """The control: without it, a test that passes after the step proves nothing."""
    model, params = _walked()
    assert _resolved(params) == len(params) == 3


def test_tape_resolves_every_slot_after_an_optimizer_step():
    """THE regression. Fails on a tree where the registry does not follow the value."""
    model, params = _walked()
    opt = optim_mod.AdamW(params, lr=1e-3, weight_decay=0.0, clip_norm=0.0)
    for t in params.values():
        t.grad = np.ones_like(t.value.arr)
    opt.step()
    moved = params.rebind()

    assert moved == 3, "rebind did not put the optimizer's tensors back into the model"
    assert _resolved(params) == 3, (
        f"the tape resolves {_resolved(params)} of 3 walked slots after one step; "
        f"every unresolved slot is a weight the next backward cannot reach")


def test_the_step_actually_moved_the_weights():
    """A registry that follows a value nothing changed would pass the test above."""
    model, params = _walked()
    before = {n: params[n].value.arr.copy() for n in params}
    opt = optim_mod.AdamW(params, lr=1e-3, weight_decay=0.0, clip_norm=0.0)
    for t in params.values():
        t.grad = np.ones_like(t.value.arr)
    opt.step()
    params.rebind()
    for n in params:
        assert not np.array_equal(before[n], getattr(model, n).arr)


def test_registration_follows_a_bare_value_replacement():
    """The seam itself, without the optimizer: `checkpoint.load_adapter` and `free` write
    a leaf's value the same way, and both would strand the key."""
    raw = _Handle(np.zeros((2, 2)))
    leaf = ag.parameter(raw)
    fresh = _Handle(np.ones((2, 2)))
    leaf.value = fresh
    assert ag.parameter_for(fresh) is leaf
    assert ag.parameter_for(raw) is None, (
        "the handle the optimizer replaced must stop resolving, or a model that skipped "
        "`rebind()` would look healthy while standing still")


def test_an_unregistered_tensor_is_not_captured_by_the_setter():
    """The negative control: the setter re-keys the leaf it owns and nothing else."""
    a, b = _Handle(np.zeros((2, 2))), _Handle(np.zeros((2, 2)))
    leaf = ag.parameter(a)
    plain = ag.Tensor(b)
    plain.value = _Handle(np.ones((2, 2)))
    assert ag.parameter_for(plain.value) is None
    assert ag.parameter_for(leaf.value) is leaf


if __name__ == "__main__":       # runnable without pytest, which no host here installs
    sys.exit(pytest.main([__file__, "-v"]) if hasattr(pytest, "main") else 1)
