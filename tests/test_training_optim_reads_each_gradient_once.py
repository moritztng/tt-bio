"""AdamW reads each gradient off the card once per pass, and the update is the same as before.

The clip is global, so `grad_norm` has to see every gradient before any is applied. It used to
read them all and then `step` / `clip_and_accumulate` read them all again. Now the norm keeps
what it read and the update consumes it. No card: `to_host` / `to_device` are patched to
numpy, with a counter on gradient reads, and the old behaviour is reproduced by a `grad_norm`
that reads every gradient a second time into `keep`, which is what the update used to do.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from tt_bio.train import optim  # noqa: E402


class _DType:
    name = "bfloat16"


class _Value:
    """A device weight: an array plus the dtype and device the optimizer asks for."""

    dtype = _DType()

    def __init__(self, arr):
        self.arr = np.asarray(arr, np.float32)

    def device(self):
        return None


class _Grad:
    def __init__(self, arr):
        self.arr = np.asarray(arr, np.float32)


class _Param:
    def __init__(self, arr):
        self.value, self.grad = _Value(arr), None


@pytest.fixture
def reads(monkeypatch):
    count = {"grad": 0}

    def to_host(t, dtype=None):
        if isinstance(t, _Grad):
            count["grad"] += 1
        return (t.arr if isinstance(t, (_Grad, _Value)) else np.asarray(t)).copy()

    monkeypatch.setattr(optim, "to_host", to_host)
    monkeypatch.setattr(optim, "to_device", lambda arr, dev, dtype=None: _Value(arr))
    return count


def _run(per_sample, old, monkeypatch):
    rng = np.random.default_rng(0)
    shapes = {"a": (4, 8), "b": (16,), "c": (3, 3, 2)}
    params = {n: _Param(rng.standard_normal(s)) for n, s in shapes.items()}
    opt = optim.AdamW(params, lr=1e-2, clip_norm=0.5, weight_decay=0.01)
    if old:
        real = optim.AdamW.grad_norm

        def read_twice(self, disabled=(), keep=None):
            # The previous code: the norm reads every gradient, then the update reads it again.
            gnorm = real(self, disabled)
            if keep is not None:
                keep.update((n, self._grad(n, t)) for n, t in self.params.items()
                            if t.grad is not None and n not in set(disabled))
            return gnorm

        monkeypatch.setattr(optim.AdamW, "grad_norm", read_twice)
    for step in range(3):
        for sample in range(2 if per_sample else 1):
            for n, p in params.items():
                p.grad = None if (n == "c" and sample == 1) else _Grad(
                    3.0 * rng.standard_normal(shapes[n]))
            if per_sample:
                opt.clip_and_accumulate(disabled={"b"} if sample == 1 else ())
        opt.step(disabled=() if per_sample else {"c"} if step == 2 else ())
        opt.zero_grad()
        if per_sample:
            opt.accum.clear(); opt.participation.clear(); opt.accum_count = 0
    return opt


@pytest.mark.parametrize("per_sample", [False, True], ids=["per-batch", "per-sample"])
def test_each_gradient_is_read_once_and_the_update_is_unchanged(per_sample, reads, monkeypatch):
    new = _run(per_sample, old=False, monkeypatch=monkeypatch)
    once = reads["grad"]
    reads["grad"] = 0
    with monkeypatch.context() as m:
        ref = _run(per_sample, old=True, monkeypatch=m)
    assert reads["grad"] == 2 * once, (reads["grad"], once)
    for n in new.master:
        assert np.array_equal(new.master[n], ref.master[n]), n
        assert np.array_equal(new.exp_avg[n], ref.exp_avg[n]), n
        assert np.array_equal(new.exp_avg_sq[n], ref.exp_avg_sq[n]), n
