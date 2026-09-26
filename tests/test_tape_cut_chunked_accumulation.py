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


# --- the sample axis: S replicates through ONE call of the shared stage -----------------
#
# `of3t-p10batch`'s lever, at unit scale and in the same file, because it is the same question
# about the same tape: a shape where several replicates read one shared node. The chunk cut
# above splits the axis in TIME; this splits it in the BATCH DIMENSION. Both have to give the
# per-replicate loop's gradient back, and the one that is easy to get wrong is the shared
# weight's, because batching turns its gradient into a reduction over the sample axis.
#
# The graph mirrors `OF3DiffusionModule._denoise_samples`: a per-replicate stage before the
# shared one (`_pre_dit`), the shared stage run once for the stack (the 24-block DiT), and a
# per-replicate stage after it reading its own slice (`_post_dit`).


def _pre(h, k):
    """The per-replicate stage before the batched one."""
    return ag.scale(h, 1.0 + 0.1 * k)


def _post(o, k):
    """The per-replicate stage after it. A distinct factor per k, so a slice that lands on the
    wrong replicate is visible in the gradient rather than cancelling out."""
    return ag.scale(o, 1.0 + 0.5 * k)


def test_a_batched_sample_axis_matches_the_per_replicate_loop(dev):
    ks = list(range(4))
    x, w1, w2 = _build(dev, seed=3)

    with ag.tape():
        h = ag.matmul(x, w1)
        roots = [_post(ag.matmul(_pre(h, k), w2), k) for k in ks]
    ag.backward(roots, _seeds(dev, roots, ks))
    loop_w1, loop_w2 = _grads(w1, w2)
    loop_roots = [_host(r) for r in roots]
    _clear(w1, w2)

    with ag.tape():
        h = ag.matmul(x, w1)
        hb = T.stack_samples([_pre(h, k) for k in ks])     # [S, N, N]
        ob = ag.matmul(hb, w2)                              # ONE matmul for the whole axis
        roots_b = [_post(ob[k:k + 1], k) for k in ks]
    ag.backward(roots_b, _seeds(dev, roots_b, ks))
    batch_w1, batch_w2 = _grads(w1, w2)

    for k in ks:
        torch.testing.assert_close(_host(roots_b[k]), loop_roots[k], rtol=2e-3, atol=2e-3)
    # w2 is the shared weight: batched, its gradient is a sum over the sample axis, which is
    # `_reduce_to`'s job and the one thing a hand-written batched backward gets wrong.
    torch.testing.assert_close(batch_w2, loop_w2, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(batch_w1, loop_w1, rtol=2e-3, atol=2e-3)


def test_every_replicate_must_read_its_own_slice(dev):
    """The control. Give every replicate slice 0 and the gradient has to come out wrong --
    otherwise the test above would pass on a graph where the sample axis never carried
    anything, which is exactly how a batched port agrees with its control for the wrong
    reason."""
    ks = list(range(4))
    x, w1, w2 = _build(dev, seed=3)

    with ag.tape():
        h = ag.matmul(x, w1)
        roots = [_post(ag.matmul(_pre(h, k), w2), k) for k in ks]
    ag.backward(roots, _seeds(dev, roots, ks))
    loop_w2 = _grads(w1, w2)[1]
    _clear(w1, w2)

    with ag.tape():
        h = ag.matmul(x, w1)
        hb = T.stack_samples([_pre(h, k) for k in ks])
        ob = ag.matmul(hb, w2)
        roots_b = [_post(ob[0:1], k) for k in ks]           # every replicate reads slice 0
    ag.backward(roots_b, _seeds(dev, roots_b, ks))
    wrong_w2 = _grads(w1, w2)[1]

    assert not torch.allclose(wrong_w2, loop_w2, rtol=2e-3, atol=2e-3), (
        "reading one slice for all four replicates changed nothing, so the slice index is not "
        "reaching the gradient and the test above proves nothing")
