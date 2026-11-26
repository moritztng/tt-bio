"""shard=True AdamW (ZeRO-1 over the launcher's ranks) steps the same model as the replicated one.

No card: to_host/to_device are numpy fakes. Two ranks run as threads over a real /dev/shm-style
rendezvous (tmp_path), each with its own gradient; the reference is one unsharded AdamW stepping
on the rank sum. Masters must agree to fp32 rounding (the global norm is summed in a different
order), be bit-identical across ranks, and each rank holds moments only for what it owns. The
broken arm (one rank skips the reduce) must not match. Every case runs twice: plain AdamW, and
with Muon taking the 2-D weights (each rank orthogonalises the whole matrices it owns).
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


# Tensor-aligned edges put a, b, c on rank 0 and d, e on rank 1: one Muon matrix per rank.
SHAPES = {"a": (4, 8), "b": (16,), "c": (3, 3, 2), "d": (40,), "e": (8, 6)}
MUON = {"adam": None, "muon": optim.Muon()}


@pytest.fixture(params=sorted(MUON))
def muon(request):
    return MUON[request.param]


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


def _run(tmp_path, broken=False, steps=3, muon=None, drop=None):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    out = {}

    def rank(r):
        ax = L.ProcessAxis(name="dp", device_ids=(0, 1), dp_rank=r, run=str(tmp_path))
        opt = optim.AdamW(_params(), lr=1e-2, clip_norm=0.5, weight_decay=0.01,
                          data_parallel=ax, shard=True, muon=muon)
        for s in range(steps):
            g = _grads(r, s)
            if broken and r == 1:
                g = {n: v * 2 for n, v in g.items()}   # a rank stepping on a different gradient
            if drop is not None and s >= 2:
                if drop == "absent":
                    del g["b"]
                else:
                    g["b"] = np.zeros_like(g["b"])
            opt.step(replicas={n: [v] for n, v in g.items()})
        out[r] = opt

    ts = [threading.Thread(target=rank, args=(r,)) for r in (0, 1)]
    [t.start() for t in ts]
    [t.join(timeout=60) for t in ts]
    ref = optim.AdamW(_params(), lr=1e-2, clip_norm=0.5, weight_decay=0.01, muon=muon)
    for s in range(steps):
        g0, g1 = _grads(0, s), _grads(1, s)
        for n, p in ref.params.items():
            p.grad = g0[n] + g1[n]
        ref.step()
    return out, ref


def test_sharded_step_matches_replicated_and_ranks_agree(tmp_path, muon):
    out, ref = _run(tmp_path, muon=muon)
    for n in SHAPES:
        assert np.array_equal(out[0].master[n], out[1].master[n])
        np.testing.assert_allclose(out[0].master[n], ref.master[n], rtol=1e-5, atol=1e-6)
    owned = [set(dict.keys(o.exp_avg)) | set(dict.keys(o.momentum_buffer))
             for o in (out[0], out[1])]
    assert owned[0] and owned[1] and not (owned[0] & owned[1])
    assert owned[0] | owned[1] == set(SHAPES)
    if muon is not None:
        bufs = [set(dict.keys(o.momentum_buffer)) for o in (out[0], out[1])]
        assert bufs[0] == {"a"} and bufs[1] == {"e"}       # each rank ran Newton-Schulz on its own
        assert not any(n in o.exp_avg for o in (out[0], out[1]) for n in "ae")


def test_a_rank_on_a_different_gradient_is_caught(tmp_path, muon):
    out, ref = _run(tmp_path, broken=True, muon=muon)
    assert any(not np.allclose(out[0].master[n], ref.master[n], rtol=1e-5, atol=1e-6)
               for n in SHAPES)


def test_a_sharded_checkpoint_is_complete_and_resumes_bit_identically(tmp_path, monkeypatch, muon):
    """Every rank saves the same complete file (moments gathered), and two fresh sharded ranks
    that load it and step once land on the same masters as the uninterrupted run."""
    from tt_bio.train import checkpoint
    monkeypatch.setattr(checkpoint, "to_device", lambda arr, dev, dtype=None: _Value(arr))
    full, _ = _run(tmp_path / "full", steps=3, muon=muon)
    res = {}

    def leg(r, phase):
        ax = L.ProcessAxis(name="dp", device_ids=(0, 1), dp_rank=r, run=str(tmp_path / phase))
        params = _params()
        opt = optim.AdamW(params, lr=1e-2, clip_norm=0.5, weight_decay=0.01,
                          data_parallel=ax, shard=True, muon=muon)
        if phase == "a":
            for s in range(2):
                opt.step(replicas={n: [v] for n, v in _grads(r, s).items()})
            checkpoint.save_adapter(tmp_path / f"ck{r}.safetensors", opt)
        else:
            checkpoint.load_adapter(tmp_path / "ck0.safetensors", params, None, opt=opt)
            held = set(dict.keys(opt.exp_avg)) | set(dict.keys(opt.momentum_buffer))
            assert held < set(SHAPES)                           # unowned state dropped
            opt.step(replicas={n: [v] for n, v in _grads(r, 2).items()})
        res[(phase, r)] = opt

    for phase in ("a", "b"):
        (tmp_path / phase).mkdir()
        ts_ = [threading.Thread(target=leg, args=(r, phase)) for r in (0, 1)]
        [t.start() for t in ts_]
        [t.join(timeout=60) for t in ts_]
    from safetensors.numpy import load_file
    blob, blob1 = (load_file(str(tmp_path / f"ck{r}.safetensors")) for r in (0, 1))
    assert sorted(blob) == sorted(blob1) and all(np.array_equal(blob[k], blob1[k]) for k in blob)
    fam = lambda n: "momentum_buffer" if muon is not None and n in "ae" else "exp_avg"
    assert all(np.abs(blob[f"{fam(n)}{checkpoint._SPLIT}{n}"]).sum() > 0 for n in SHAPES)
    for n in SHAPES:
        assert np.array_equal(res[("b", 0)].master[n], full[0].master[n])
        assert np.array_equal(res[("b", 1)].master[n], full[0].master[n])


def test_an_absent_gradient_sends_zeros_not_the_reused_slot(tmp_path, muon):
    """The send slot is rewritten in place, so it still holds an earlier step's bytes: a
    parameter with no gradient this step must be sent as zeros, exactly like an explicit zero."""
    gone, _ = _run(tmp_path / "absent", steps=5, muon=muon, drop="absent")
    zero, _ = _run(tmp_path / "zero", steps=5, muon=muon, drop="zero")
    for n in SHAPES:
        assert np.array_equal(gone[0].master[n], zero[0].master[n]), n
