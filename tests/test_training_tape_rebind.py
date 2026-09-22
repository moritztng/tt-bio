"""The optimizer/tape contract: a stepped leaf is still a leaf the tape can resolve.

``autograd._PARAMS`` maps the identity of the RAW handle the model holds to the leaf that
owns it, because that is the only question a taped call can ask. Three callers in this tree
legitimately replace a leaf's handle: ``AdamW.step`` (``train/optim.py``),
``checkpoint.load_adapter`` (``train/checkpoint.py``) and ``autograd.Tensor.free``'s L1
eviction. If the registry does not follow, ``parameter_for`` returns None at every call site
from that moment on and the next backward reaches zero parameters, with real weights, a real
optimizer and a loss curve that still falls.

**What this does NOT claim.** Main's own shipped recipe does not reach the defect. It goes
``train/recipes.py`` -> ``lora.trainable`` -> ``lora._Substitute``, and ``_Substitute`` hands
the forward the LEAF, so a taped call there never asks ``_PARAMS`` to resolve a raw handle.
Measured on a card at 9e17ad418, six steps, the registry resolving 0 of 5 leaves and the run
training perfectly: ``perf/d126_main_reach/out/weights/reach.json``. What this protects is the
PUBLISHED seam -- ``parameter`` is in ``autograd.__all__``, and a Tier-2 caller writing its own
full-weight loop over it reaches exactly the stranded key.

No card. The device handles are stand-ins and the transfers are numpy, which is deliberate:
this is a contract between the optimizer and the tape, it is the cheapest regression signal
there is, and a test that needs a card is a test a gate skips.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ttnn = pytest.importorskip("ttnn")      # imported by tt_bio.autograd; no device is opened

from tt_bio import autograd as ag                                   # noqa: E402
from tt_bio.train import optim as optim_mod                         # noqa: E402


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


@pytest.fixture(autouse=True)
def _host_transfers(monkeypatch):
    """Host<->'device' as numpy. ``to_device`` mints a NEW handle, which is the whole point:
    the defect is that the registry stays keyed on the one it replaced."""
    monkeypatch.setattr(optim_mod, "to_host",
                        lambda t, dtype=None: (t.arr if isinstance(t, _Handle)
                                               else np.asarray(t)).astype(
                                                   np.float32 if dtype is None else dtype))
    monkeypatch.setattr(optim_mod, "to_device",
                        lambda arr, device, dtype=None, layout=None: _Handle(arr))
    yield
    ag.forget_parameters()


def _registered(n_tensors=3, dim=4):
    """A Tier-2 caller's parameter set: raw handles the model holds, declared trainable."""
    rng = np.random.default_rng(20260921)
    raw = {f"w{i}": _Handle(rng.standard_normal((dim, dim))) for i in range(n_tensors)}
    return raw, {name: ag.parameter(h) for name, h in raw.items()}


def _resolved(params):
    """How many leaves a taped call could still hand a gradient to.

    Through ``_param_on_tape``, which is what ``_on_tape`` and ``_differentiating`` actually
    call, rather than through the registry dict: a test that reads ``_PARAMS`` directly stops
    agreeing with the tape the first time the tape changes.
    """
    return sum(1 for t in params.values() if ag._param_on_tape(t.value) is t)


def test_registered_handles_resolve_before_any_step():
    """The control. Without it, a test that passes after the step proves nothing."""
    raw, params = _registered()
    assert _resolved(params) == len(params) == 3


def test_a_taped_call_resolves_the_stepped_handle():
    """THE regression. Fails on a tree where the registry does not follow the value."""
    raw, params = _registered()
    opt = optim_mod.AdamW(params, lr=1e-3, weight_decay=0.0, clip_norm=0.0)
    for t in params.values():
        t.grad = np.ones_like(t.value.arr)
    opt.step()
    assert _resolved(params) == 3, (
        f"a taped call resolves {_resolved(params)} of 3 parameters after one step; "
        f"every unresolved one is a weight the next backward cannot reach")


def test_the_step_actually_moved_the_weights():
    """A registry that followed a value nothing changed would pass the test above."""
    raw, params = _registered()
    before = {n: t.value.arr.copy() for n, t in params.items()}
    opt = optim_mod.AdamW(params, lr=1e-3, weight_decay=0.0, clip_norm=0.0)
    for t in params.values():
        t.grad = np.ones_like(t.value.arr)
    opt.step()
    for n, t in params.items():
        assert not np.array_equal(before[n], t.value.arr)


def test_registration_follows_a_bare_value_replacement():
    """The seam itself, without the optimizer.

    ``checkpoint.load_adapter`` writes a checkpoint's tensor into ``t.value`` and ``free``
    writes an L1-to-DRAM copy into it. Both are this line, and both would strand the key.
    """
    raw = _Handle(np.zeros((2, 2)))
    leaf = ag.parameter(raw)
    fresh = _Handle(np.ones((2, 2)))
    leaf.value = fresh
    assert ag.parameter_for(fresh) is leaf
    assert ag.parameter_for(raw) is None, (
        "the handle that was replaced must stop resolving, or a second `parameter(raw)` "
        "mints a rival leaf over the same weight")


def test_an_unregistered_tensor_is_not_captured_by_the_setter():
    """The negative control: the setter re-keys the leaf it owns and nothing else."""
    a, b = _Handle(np.zeros((2, 2))), _Handle(np.zeros((2, 2)))
    leaf = ag.parameter(a)
    plain = ag.Tensor(b)
    plain.value = _Handle(np.ones((2, 2)))
    assert ag.parameter_for(plain.value) is None
    assert ag.parameter_for(leaf.value) is leaf


def test_value_still_reads_back_what_was_written():
    """The property must not change what ``value`` MEANS, only what writing it also does."""
    h = _Handle(np.zeros((2, 2)))
    t = ag.Tensor(h)
    assert t.value is h
    fresh = _Handle(np.ones((2, 2)))
    t.value = fresh
    assert t.value is fresh
    # `__getattr__` forwards the rest to the handle, and it reads the SLOT: forwarding
    # through the property would recurse in the instant before the slot is set.
    assert t.device() == "fake-device"
    assert t.shape == (2, 2)


if __name__ == "__main__":       # runnable without pytest, which no host here installs
    sys.exit(pytest.main([__file__, "-v"]))
