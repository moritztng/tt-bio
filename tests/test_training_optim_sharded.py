"""shard=True AdamW (ZeRO-1 over the launcher's ranks) steps the same model as the replicated one.

No card: to_host/to_device are numpy fakes. Two ranks run as threads over a real /dev/shm-style
rendezvous (tmp_path), each with its own gradient; the reference is one unsharded AdamW stepping
on the rank sum. Masters must agree to fp32 rounding (the global norm is summed in a different
order), be bit-identical across ranks, and each rank holds moments only for what it owns. The
broken arm (one rank skips the reduce) must not match.
"""
import sys, threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from tt_bio.train import optim, launcher as L  # noqa: E402


class _DType:
    name = "bfloat16"


class _Value:
    dtype = _DType()

    def __init__(self, arr):
        self.arr = np.asarray(arr, np.float32)

    def device(self):
        return None


class _Param:
    def __init__(self, arr):
        self.value, self.grad = _Value(arr), None


SHAPES = {"a": (4, 8), "b": (16,), "c": (3, 3, 2), "d": (40,)}


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(optim, "to_host", lambda t, dtype=None: np.asarray(
        t.arr if isinstance(t, _Value) else t, np.float32).copy())
    monkeypatch.setattr(optim, "to_device", lambda arr, dev, dtype=None: _Value(arr))


def _params():
    rng = np.random.default_rng(0)
    return {n: _Param(rng.standard_normal(s)) for n, s in SHAPES.items()}


def _grads(rank, step):
    rng = np.random.default_rng(100 * step + rank)
    return {n: rng.standard_normal(s).astype(np.float32) for n, s in SHAPES.items()}


def _run(tmp_path, broken=False, steps=3):
    out = {}

    def rank(r):
        ax = L.ProcessAxis(name="dp", device_ids=(0, 1), dp_rank=r, run=str(tmp_path))
        opt = optim.AdamW(_params(), lr=1e-2, clip_norm=0.5, weight_decay=0.01,
                          data_parallel=ax, shard=True)
        for s in range(steps):
            g = _grads(r, s)
            if broken and r == 1:
                g = {n: v * 2 for n, v in g.items()}   # a rank stepping on a different gradient
            opt.step(replicas={n: [v] for n, v in g.items()})
        out[r] = opt

    ts = [threading.Thread(target=rank, args=(r,)) for r in (0, 1)]
    [t.start() for t in ts]
    [t.join(timeout=60) for t in ts]
    ref = optim.AdamW(_params(), lr=1e-2, clip_norm=0.5, weight_decay=0.01)
    for s in range(steps):
        g0, g1 = _grads(0, s), _grads(1, s)
        for n, p in ref.params.items():
            p.grad = g0[n] + g1[n]
        ref.step()
    return out, ref


def test_sharded_step_matches_replicated_and_ranks_agree(tmp_path):
    out, ref = _run(tmp_path)
    for n in SHAPES:
        assert np.array_equal(out[0].master[n], out[1].master[n])
        np.testing.assert_allclose(out[0].master[n], ref.master[n], rtol=1e-5, atol=1e-6)
    owned = [set(dict.keys(o.exp_avg)) for o in (out[0], out[1])]
    assert owned[0] and owned[1] and not (owned[0] & owned[1])
    assert owned[0] | owned[1] == set(SHAPES)


def test_a_rank_on_a_different_gradient_is_caught(tmp_path):
    out, ref = _run(tmp_path, broken=True)
    assert any(not np.allclose(out[0].master[n], ref.master[n], rtol=1e-5, atol=1e-6)
               for n in SHAPES)


def test_saving_a_sharded_optimizer_refuses(tmp_path):
    from tt_bio.train import checkpoint
    out, _ = _run(tmp_path, steps=1)
    with pytest.raises(NotImplementedError):
        checkpoint.save_adapter(tmp_path / "x.safetensors", out[0])
