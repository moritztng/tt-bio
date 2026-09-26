"""The replicate axis: the split, and the reduce that makes two ranks agree with one.

Card-free. The property under test is arithmetic, not hardware: N ranks each differentiating
a disjoint slice of the replicates and summing must give the gradient one rank differentiating
all of them gives. A test that only checked the split would pass against a reduce that drops
a rank.
"""
import inspect
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


SEED = 20260921


def _one_chip(n, shape=(7, 3)):
    """What a single chip differentiates: replicate i, noised by its own index."""
    return {i: rdp.replicate_noise(SEED, i, shape) for i in range(n)}


def _sharded_walk(n, chunk, world, shape=(7, 3)):
    """The harness's real walk: replicate 0 primed on every rank, the rest in chunks of
    `chunk` starting at 1, and the chunk STARTS round-robined over ranks."""
    starts = list(range(1, n, chunk))
    seen = {}
    for rank in range(world):
        seen[0] = rdp.replicate_noise(SEED, 0, shape)      # the primed replicate, every rank
        for ci in [starts[i] for i in rdp.shard(len(starts), world, rank)]:
            for k in range(min(chunk, n - ci)):
                gi = ci + k
                assert gi not in seen or gi == 0, f"replicate {gi} ran on two ranks"
                seen[gi] = rdp.replicate_noise(SEED, gi, shape)
    return seen


def _sharded_walk_draw_order(n, chunk, world, shape=(7, 3)):
    """The DEFECT this module exists to prevent, kept as a control: one generator per rank,
    consumed in the order that rank happens to execute. Identical on one chip."""
    import numpy as np
    seen = {}
    starts = list(range(1, n, chunk))
    for rank in range(world):
        rng = np.random.default_rng(SEED)
        seen[0] = rng.standard_normal(shape).astype("float32")
        for ci in [starts[i] for i in rdp.shard(len(starts), world, rank)]:
            for k in range(min(chunk, n - ci)):
                seen[ci + k] = rng.standard_normal(shape).astype("float32")
    return seen


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("chunk", [4, 8, 12])
def test_the_sharded_walk_differentiates_exactly_what_one_chip_does(world, chunk):
    """48 replicates total across all chips, each exactly once, each the structure replicate
    i is on one chip. This is the NO-CHEAT check as much as a correctness one: an arm that
    ran 12 replicates and called it 48 would fail here."""
    one = _one_chip(48)
    many = _sharded_walk(48, chunk, world)
    assert sorted(many) == list(range(48))
    for i in range(48):
        np.testing.assert_array_equal(many[i], one[i])


def test_the_draw_order_scheme_agrees_on_one_chip_and_breaks_under_a_shard():
    """The control. Without it this suite would pass against the defective scheme too, which
    is exactly how the defect survived: on one chip the two schemes are the same walk."""
    one = _one_chip(48)
    assert set(_sharded_walk_draw_order(48, 4, 1)) == set(one)     # reaches every index
    broken = _sharded_walk_draw_order(48, 4, 2)
    mismatched = [i for i in range(48)
                  if not np.array_equal(broken[i], _sharded_walk_draw_order(48, 4, 1)[i])]
    assert mismatched, "the control did not reproduce the defect, so it proves nothing"
    # And the sharp edge: two DIFFERENT replicates end up with identical noise.
    dupes = [(i, j) for i in range(48) for j in range(i + 1, 48)
             if np.array_equal(broken[i], broken[j])]
    assert dupes, "expected the draw-order scheme to duplicate noise across ranks"


def _one_chip_gradients(n_rep):
    """What one chip's step produces: every replicate's contribution, plus the prefix's."""
    diffusion = sum(_rep_contribution(i) for i in range(n_rep))
    return {"diffusion.w": diffusion, "trunk.w": _prefix_contribution(diffusion)}


def _rep_contribution(i):
    return np.full(4, float(i + 1), np.float32)


def _prefix_contribution(summed_cotangent):
    """The prefix backward: a function of the SUMMED cotangent, so every rank computing it
    from the reduced sum gets the same answer."""
    return summed_cotangent * 2.0


def _step(world, n_rep, reduce_before_prefix):
    """The step as each rank runs it, with a perfect in-process reduce standing in for the
    /dev/shm one (which `test_reduce_grads_sums_across_ranks...` covers for real)."""
    prime = 0
    mine = {r: [i for i in rdp.shard(n_rep - 1, world, r)] for r in range(world)}
    # each rank: its own replicates' diffusion gradient, and its own cut cotangent
    diff = {r: sum((_rep_contribution(1 + i) for i in mine[r]), np.zeros(4, np.float32))
            for r in range(world)}
    cot = {r: diff[r].copy() for r in range(world)}
    summed_cot = sum(cot.values()) + _rep_contribution(prime)   # reduce_cut, before anything

    if reduce_before_prefix:
        reduced = sum(diff.values())                            # replicate half only
        diff = {r: reduced.copy() for r in range(world)}
    out = {}
    for r in range(world):
        # the prefix backward, on EVERY rank, from the summed cotangent. It also walks the
        # primed replicate, so it adds that replicate's diffusion contribution too.
        d = diff[r] + _rep_contribution(prime)
        out[r] = {"diffusion.w": d, "trunk.w": _prefix_contribution(summed_cot)}
    if not reduce_before_prefix:
        reduced = sum(o["diffusion.w"] for o in out.values())   # the WRONG order
        for r in range(world):
            out[r] = {"diffusion.w": reduced, "trunk.w": out[r]["trunk.w"]}
    return out


@pytest.mark.parametrize("world", [1, 2, 4])
def test_reducing_before_the_prefix_backward_matches_one_chip(world):
    """The ordering IS the correctness argument, not a performance detail."""
    one = _one_chip_gradients(12)
    for rank, got in _step(world, 12, reduce_before_prefix=True).items():
        for name in ("diffusion.w", "trunk.w"):
            np.testing.assert_allclose(got[name], one[name], rtol=0, atol=0,
                                       err_msg=f"rank {rank} parameter {name}")


def test_reducing_after_the_prefix_backward_counts_the_primed_replicate_world_times():
    """The control. It is exact at world 1, which is how the wrong order would ship."""
    one = _one_chip_gradients(12)
    np.testing.assert_array_equal(
        _step(1, 12, reduce_before_prefix=False)[0]["diffusion.w"], one["diffusion.w"])
    for world in (2, 4):
        got = _step(world, 12, reduce_before_prefix=False)[0]["diffusion.w"]
        over = got - one["diffusion.w"]
        np.testing.assert_array_equal(over, _rep_contribution(0) * (world - 1))


@pytest.mark.parametrize("world", [2, 4])
def test_every_rank_ends_the_step_with_identical_gradients(world):
    """What makes the master weights bit-identical across ranks with no broadcast: not that
    the ranks agree with N=1 bit for bit (the reduce sums in fp32 where N=1 never leaves the
    card), but that they agree with EACH OTHER."""
    ranks = _step(world, 12, reduce_before_prefix=True)
    first = ranks[0]
    for r, got in ranks.items():
        for name in first:
            np.testing.assert_array_equal(got[name], first[name], err_msg=f"rank {r} {name}")


def test_reduce_grads_over_a_name_set_leaves_the_rest_untouched():
    """The prefix's weights must not cross the wire, and `names` is how. A parameter outside
    the set keeps whatever the rank computed, which is the whole point: every rank computed
    the same thing."""
    assert "names" in inspect.signature(rdp.reduce_grads).parameters


def test_replicate_noise_is_a_pure_function_of_seed_and_index():
    a = rdp.replicate_noise(SEED, 17, (5, 3))
    assert np.array_equal(a, rdp.replicate_noise(SEED, 17, (5, 3)))
    assert not np.array_equal(a, rdp.replicate_noise(SEED, 18, (5, 3)))
    assert not np.array_equal(a, rdp.replicate_noise(SEED + 1, 17, (5, 3)))
    assert a.shape == (5, 3) and a.dtype == np.float32


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
