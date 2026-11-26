"""`train.folds`: the gradient of a load-time weight fold reaches the upstream parameters. Host-only.

A toy build folds the way RFD4 does: a norm weight multiplied into a projection inside a closure, a QKV
split by output column, a second packed copy of one slice, and a zero bias the checkpoint lacks. Device
weights are host `ttnn` tiles.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from tt_bio.train.folds import Folds  # noqa: E402

KEYS = ("gamma", "q", "k", "v", "packed")


def _state(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"to_gamma.weight": torch.randn(8, 6, generator=g), "norm_cond.weight": torch.randn(6, generator=g),
            "to_qkv.weight": torch.randn(12, 6, generator=g)}


def _build(sd):
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)  # noqa: E731
    w = sd["norm_cond.weight"].float()
    gw = lambda m: (m.float() * w).t()  # noqa: E731  (a closure, as AdaptiveNorm folds)
    qkv = sd["to_qkv.weight"]
    return {"gamma": up(gw(sd["to_gamma.weight"])), "q": up(qkv[:4].t()), "k": up(qkv[4:8].t()),
            "v": up(qkv[8:].t()), "packed": up(qkv[:4].t().contiguous()), "zero_bias": up(torch.zeros(6))}


def _upstream_grads(st, G):
    ref = {k: v.double().clone().requires_grad_(True) for k, v in st.items()}
    q, w = ref["to_qkv.weight"], ref["norm_cond.weight"]
    loss = ((ref["to_gamma.weight"] * w).t() * G["gamma"]).sum() + (q[:4].t() * G["q"]).sum() \
        + (q[4:8].t() * G["k"]).sum() + (q[8:].t() * G["v"]).sum() + (q[:4].t() * G["packed"]).sum()
    loss.backward()
    return {k: v.grad for k, v in ref.items()}


def _setup():
    st = _state()
    f = Folds(st)
    with f.recording() as sd:
        m = _build(sd)
    gen = torch.Generator().manual_seed(1)
    G = {k: torch.randn(tuple(m[k].shape), generator=gen, dtype=torch.float64) for k in KEYS}
    return st, f, m, G


def test_fold_vjp_equals_the_upstream_gradient():
    st, f, m, G = _setup()
    f.backward([(m[k], G[k]) for k in KEYS])
    for n, r in _upstream_grads(st, G).items():
        got = f.leaves[n].grad.double()
        assert torch.allclose(got, r, rtol=1e-5, atol=1e-5), f"{n}: {(got - r).abs().max():.2e}"
    assert f.leaves["norm_cond.weight"].grad.abs().sum() > 0  # the closure's parameter trains


def test_the_recorded_build_is_the_serving_build_bit_for_bit():
    st, f, m, _ = _setup()
    served = _build(st)
    for k in m:
        assert torch.equal(ttnn.to_torch(m[k]), ttnn.to_torch(served[k])), k


def test_a_constant_and_an_orphan_are_named_and_refused():
    _, f, m, _ = _setup()
    m["on_device"] = ttnn.from_torch(torch.ones(4), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)  # outside recording
    assert f.constants(m) == ["zero_bias"]
    assert f.orphans(m) == ["on_device"]
    assert f.unreached() == []
    with pytest.raises(KeyError):
        f.backward([(m["zero_bias"], torch.ones(6))])


def test_a_dropped_duplicate_fails_the_comparison():
    """Broken arm: a VJP that skips the packed copy must disagree with the upstream gradient."""
    st, f, m, G = _setup()
    f.backward([(m[k], G[k]) for k in KEYS if k != "packed"])
    ref = _upstream_grads(st, G)["to_qkv.weight"]
    assert not torch.allclose(f.leaves["to_qkv.weight"].grad.double(), ref, rtol=1e-5, atol=1e-5)
