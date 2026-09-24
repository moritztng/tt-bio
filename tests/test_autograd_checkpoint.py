"""`autograd.checkpoint` on a segment with several outputs: one recompute, the plain gradient,
and a tape that refcounting frees on its own.

No card. Handles are numpy stand-ins and the three ttnn verbs `backward` and `add_grad` call
(`add`, `typecast`, `ones_like`) are patched to numpy, so what is tested is the tape's
bookkeeping, not arithmetic. The device side of the same claims is `perf/bcx_stack/stack.py
ckbits` (an AF2 block checkpointed matches it uncheckpointed bit for bit) and `gcdiag`.
"""

import contextlib
import gc
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

np = pytest.importorskip("numpy")
pytest.importorskip("ttnn")               # imported by tt_bio.autograd; no device is opened

from tt_bio import autograd as ag          # noqa: E402
from tt_bio import taped_ttnn              # noqa: E402


class _Handle:
    """What the tape asks a ttnn tensor. `storage` is shared by a view, like a ttnn reshape."""

    def __init__(self, arr, storage=None):
        self.arr = np.asarray(arr, dtype=np.float64)
        self.storage = storage if storage is not None else object()
        self.dtype = "fp"

    @property
    def shape(self):
        return self.arr.shape

    def buffer_address(self):
        return id(self.storage)

    def memory_config(self):
        raise RuntimeError("host stand-in")      # `_evict_read_parents` treats it as host


@pytest.fixture(autouse=True)
def numpy_ttnn(monkeypatch):
    monkeypatch.setattr(ag.ttnn, "add", lambda a, b: _Handle(a.arr + b.arr), raising=False)
    monkeypatch.setattr(ag.ttnn, "typecast", lambda a, dtype: a, raising=False)
    monkeypatch.setattr(ag.ttnn, "ones_like", lambda a: _Handle(np.ones_like(a.arr)), raising=False)
    monkeypatch.setattr(taped_ttnn, "recompute_scope", contextlib.nullcontext)
    yield
    ag.release_pins()


def _scale(x, c):
    return ag._tape(_Handle(c * x.value.arr), [x], lambda: (lambda g: x.add_grad(_Handle(c * g.arr))))


def _view(x):
    """A view: same storage, so `_tape` puts it in one `shares` group with its source."""
    return ag._tape(_Handle(x.value.arr, x.value.storage), [x], lambda: (lambda g: x.add_grad(g)))


def _mul(a, b):
    return ag._tape(_Handle(a.value.arr * b.value.arr), [a, b],
                    lambda: (lambda g: (a.add_grad(_Handle(g.arr * b.value.arr)),
                                        b.add_grad(_Handle(g.arr * a.value.arr)))))


CALLS = []


def _block(m, z):
    """Two outputs, and the second reads the first, as AF2's pair track reads the final MSA."""
    CALLS.append(1)
    m2 = _scale(_view(m), 2.0)
    return m2, _mul(z, _scale(m2, 3.0))


def _grads(ckpt, seed_m=True):
    m = ag.Tensor(_Handle([1.0, 2.0]), requires_grad=True)
    z = ag.Tensor(_Handle([5.0, 7.0]), requires_grad=True)
    mo, zo = ag.checkpoint(_block, m, z) if ckpt else _block(m, z)
    roots, seeds = [zo], [_Handle([1.0, -1.0])]
    if seed_m:
        roots, seeds = [mo, zo], [_Handle([0.5, 0.25])] + seeds
    ag.backward(roots, seeds)
    return m.grad.arr.copy(), z.grad.arr.copy()


@pytest.mark.parametrize("seed_m", [True, False], ids=["both-outputs-seeded", "one-output-seeded"])
def test_a_two_output_segment_recomputes_once_and_matches_the_plain_gradient(seed_m):
    CALLS.clear()
    plain = _grads(ckpt=False, seed_m=seed_m)
    assert len(CALLS) == 1
    CALLS.clear()
    ckpt = _grads(ckpt=True, seed_m=seed_m)
    assert len(CALLS) == 2, "one untaped forward plus ONE recompute"
    for p, c in zip(plain, ckpt):
        np.testing.assert_array_equal(p, c)


def test_the_recomputed_tape_is_freed_without_the_cyclic_collector():
    _grads(ckpt=True)
    gc.collect()
    was = gc.isenabled()
    gc.disable()
    try:
        _grads(ckpt=True)
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()
        # The tape: nodes, and the tensors that carry one. A segment's untaped forward also
        # groups a view with the leaf it views, and those node-less pairs are not the tape.
        left = [o for o in gc.garbage
                if isinstance(o, ag._Node) or (isinstance(o, ag.Tensor) and o.node is not None)]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        if was:
            gc.enable()
    assert not left, f"{len(left)} tape objects were only reachable through a cycle"
