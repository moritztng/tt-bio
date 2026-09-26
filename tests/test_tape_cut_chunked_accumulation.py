"""Cutting a tape at a shared node and accumulating the cut's cotangent is exact.

`perf/of3t_stepfloor/fullstep.py` runs OpenFold3's 48 diffusion replicates in chunks, because
48 of them at once wants ~170 GB of a 34.23 GB card. The chunking is only legitimate because
the diffusion loss is a sum over structures that are independent GIVEN the trunk output, so the
derivative of the sum is the sum of the per-chunk derivatives.

The mechanism it uses is `fullstep.cut`: a detached leaf over the shared node's own buffer. Each
chunk's backward stops there and accumulates, and the shared prefix is differentiated ONCE, from
the accumulated cotangent, after the last chunk. That matters for more than arithmetic -- the
trunk's ~20 s of backward is per backward ENTRY, so a chunk loop that let every chunk reach the
trunk would pay it twelve times.

This is the arithmetic half, at unit scale and in one second: the same graph differentiated whole
and in chunks has to produce the same gradient on every leaf, and the cut's accumulated cotangent
has to equal the one the whole backward puts there. The model-scale half is `fullstep.py
--grad-ab`, which grades 3,152 real parameters on a card.

The last test is the control: a cut whose chunks are NOT accumulated gets the wrong answer, so a
pass above is not the test failing to look.
"""
from __future__ import annotations

import pytest
import torch
import ttnn

from tt_bio import autograd as ag
from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device

N = 64
TILE = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)


@pytest.fixture
def dev():
    return T.get_device()


def _dev_t(dev, t):
    return ttnn.from_torch(t, device=dev, **TILE)


def _host(x):
    return ttnn.to_torch(ag._unwrap(x) if isinstance(x, ag.Tensor) else x).float()


def _cut(t):
    """`fullstep.cut`, kept in step with it by shape rather than by import: a perf harness is not
    an importable module for the test tree, and the mechanism is three lines."""
    d = ag.Tensor(ag._unwrap(t), requires_grad=True)
    d.evictable = False
    return d


def _build(dev, seed=0):
    """A shared prefix (`w1`) feeding R independent replicates (`w2`), the shape of a trunk
    feeding diffusion replicates."""
    g = torch.Generator().manual_seed(seed)
    # A model input, not a weight: a leaf the tape can read but nobody steps.
    x = ag.Tensor(_dev_t(dev, torch.randn(1, N, N, generator=g)),
                  requires_grad=False)
    w1 = ag.parameter(_dev_t(dev, torch.randn(1, N, N, generator=g)))
    w2 = ag.parameter(_dev_t(dev, torch.randn(1, N, N, generator=g)))
    return x, w1, w2


def _replicates(h, w2, ks):
    """One replicate per k, each reading the shared node `h` and the shared weight `w2`, each at
    its own scale -- the stand-in for a per-replicate noise level."""
    return [ag.scale(ag.matmul(h, w2), 1.0 + 0.5 * k) for k in ks]


def _seeds(dev, roots, ks):
    """A distinct cotangent per replicate, so a chunk that lands on the wrong root shows up."""
    return [_dev_t(dev, torch.full((1, N, N), 0.1 * (k + 1))) for k in ks]


def _grads(w1, w2):
    return _host(w1.grad), _host(w2.grad)


def _clear(*params):
    for p in params:
        p.grad = None


def test_chunked_backward_matches_the_whole_one(dev):
    """Four replicates, differentiated whole and then in chunks of two, one shared prefix."""
    ks = list(range(4))
    x, w1, w2 = _build(dev)

    with ag.tape():
        h = ag.matmul(x, w1)
        roots = _replicates(h, w2, ks)
    ag.backward(roots, _seeds(dev, roots, ks))
    whole_w1, whole_w2 = _grads(w1, w2)
    whole_h = _host(h.grad) if h.grad is not None else None
    _clear(w1, w2)

    with ag.tape():
        h2 = ag.matmul(x, w1)
    cut = _cut(h2)
    entries = 0
    for lo in range(0, len(ks), 2):
        part = ks[lo:lo + 2]
        with ag.tape():
            r = _replicates(cut, w2, part)
        ag.backward(r, _seeds(dev, r, part))
        entries += 1
    assert entries == 2, "the chunk loop did not run twice"
    assert cut.grad is not None, "no cotangent reached the cut"
    cut_cot = _host(cut.grad)
    ag.backward([h2], [cut.grad])
    chunk_w1, chunk_w2 = _grads(w1, w2)

    torch.testing.assert_close(chunk_w2, whole_w2, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(chunk_w1, whole_w1, rtol=2e-3, atol=2e-3)
    if whole_h is not None:
        torch.testing.assert_close(cut_cot, whole_h, rtol=2e-3, atol=2e-3)


def test_the_prefix_is_entered_once_however_many_chunks(dev):
    """The performance half of the same property, as a graph fact rather than a stopwatch.

    Every chunk's reverse walk must stop at the cut, so the shared prefix appears in exactly one
    traversal no matter how the replicates are split. Counted on `_reverse_topo`, which is what
    `backward` walks.
    """
    ks = list(range(4))
    x, w1, w2 = _build(dev, seed=1)
    with ag.tape():
        h = ag.matmul(x, w1)
    cut = ag.Tensor(ag._unwrap(h), requires_grad=True)
    cut.evictable = False

    prefix = {id(t) for t in ag._reverse_topo([h])}
    for k in ks:
        with ag.tape():
            r = _replicates(cut, w2, [k])
        walked = {id(t) for t in ag._reverse_topo(r)}
        assert not (walked & prefix), (
            f"chunk {k}'s backward reaches {len(walked & prefix)} node(s) of the shared prefix; "
            "the cut is not holding and the prefix would be differentiated once per chunk")
        ag.backward(r, _seeds(dev, r, [k]))
    _clear(w1, w2)


def test_dropping_the_accumulation_is_caught(dev):
    """The control. Backward the prefix from ONE chunk's cotangent instead of the sum, and the
    shared weight's gradient must come out wrong -- otherwise the test above passes on a graph
    where the accumulation never mattered."""
    ks = list(range(4))
    x, w1, w2 = _build(dev, seed=2)

    with ag.tape():
        h = ag.matmul(x, w1)
        roots = _replicates(h, w2, ks)
    ag.backward(roots, _seeds(dev, roots, ks))
    whole_w1, _ = _grads(w1, w2)
    _clear(w1, w2)

    with ag.tape():
        h2 = ag.matmul(x, w1)
    cut = _cut(h2)
    last = None
    for lo in range(0, len(ks), 2):
        part = ks[lo:lo + 2]
        cut.grad = None                      # deliberately forget the earlier chunks
        with ag.tape():
            r = _replicates(cut, w2, part)
        ag.backward(r, _seeds(dev, r, part))
        last = cut.grad
    ag.backward([h2], [last])
    partial_w1, _ = _grads(w1, w2)
    _clear(w1, w2)

    assert not torch.allclose(partial_w1, whole_w1, rtol=2e-3, atol=2e-3), (
        "dropping three quarters of the cut's cotangent changed nothing, so this file is not "
        "testing the accumulation")
