"""The replicate axis: the split, and the reduce that makes two ranks agree with one.

Card-free. The property under test is arithmetic, not hardware: N ranks each differentiating
a disjoint slice of the replicates and summing must give the gradient one rank differentiating
all of them gives. A test that only checked the split would pass against a reduce that drops
a rank.
"""
import multiprocessing as mp
import tempfile

import numpy as np
import pytest

from tt_bio.train.launcher import ProcessAxis
from tt_bio.train.mesh import Mesh
from tt_bio.train import replicate_dp as rdp


def test_shard_is_a_partition():
    for n, w in ((48, 1), (47, 2), (47, 4), (8, 4), (3, 4)):
        parts = [rdp.shard(n, w, r) for r in range(w)]
        assert sorted(i for p in parts for i in p) == list(range(n))
        assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1


def test_shard_is_round_robin_not_contiguous():
    # The sigma schedule is ordered by noise level, so a contiguous split gives the ranks
    # systematically different work and the barrier waits on the schedule, not the hardware.
    assert rdp.shard(8, 2, 0) == [0, 2, 4, 6]
    assert rdp.shard(8, 2, 1) == [1, 3, 5, 7]


def test_shard_rejects_a_rank_off_its_axis():
    with pytest.raises(ValueError):
        rdp.shard(8, 2, 2)


class _P:
    """A parameter stand-in: something with `.grad`, which is all the reduce touches."""

    def __init__(self, grad, shape=(1,)):
        self.grad = grad
        self.shape = shape


def _rank(rank, world, run, n_leaves, q):
    axis = ProcessAxis(name="dp", device_ids=tuple(range(world)),
                       mesh=Mesh({"dp": list(range(world))}), dp_rank=rank, run=run)
    # Rank r owns replicates r, r+world, ... and each contributes (i+1) to w0 and (i+1)*2 to
    # w1; w2 takes a gradient on rank 0 only, standing in for the prefix weights.
    mine = rdp.shard(n_leaves, world, rank)
    params = {
        "w0": _P(np.array([sum(i + 1 for i in mine)], np.float32)),
        "w1": _P(np.array([sum((i + 1) * 2 for i in mine)], np.float32)),
        "w2": _P(np.array([7.0], np.float32)) if rank == 0 else _P(None),
        "w3": _P(None),
    }
    rdp.reduce_grads(axis, params)
    q.put((rank, {k: (None if v.grad is None else np.asarray(v.grad).tolist())
                  for k, v in params.items()}))


def test_reduce_grads_sums_across_ranks_and_keeps_an_absent_gradient_absent():
    n = 7
    ctx = mp.get_context("spawn")
    for world in (2, 4):
        with tempfile.TemporaryDirectory() as run:
            q = ctx.Queue()
            ps = [ctx.Process(target=_rank, args=(r, world, run, n, q))
                  for r in range(world)]
            for p in ps:
                p.start()
            got = dict(q.get(timeout=120) for _ in range(world))
            for p in ps:
                p.join(timeout=30)
                assert p.exitcode == 0
        total = sum(i + 1 for i in range(n))
        for r in range(world):
            assert got[r]["w0"] == [float(total)], (world, r, got[r])
            assert got[r]["w1"] == [float(total * 2)]
            # present on one rank only, so it survives the sum at its own value
            assert got[r]["w2"] == [7.0]
            # present on NO rank: stays None. Adam with a zero gradient still moves a weight
            # through its momentum, so None and zeros are different updates.
            assert got[r]["w3"] is None
