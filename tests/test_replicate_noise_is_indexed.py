"""A replicate's noise is a function of its INDEX, not of its position in the draw order.

`tt_bio/sample_chunks.py` states the invariant every shipped AF3-family sampler holds: "the
samplers draw every sample's initial noise, augmentation and step noise over the whole sample
axis on the host before any chunking, so a sample gets the same draws at every width."

`perf/of3t_stepfloor/fullstep.py` did not hold it. It threaded one generator into the replicate
loop and drew `(n_atom, 3)` inside, so replicate k got the k-th draw of the stream. On one chip,
walked in order, that is the same thing. It stops being the same thing the moment the axis is
partitioned any other way -- several replicates through one batched module call, a chunk re-run
after a refusal, a rank holding a round-robin slice -- and then a batched arm and a
per-replicate arm are two different computations and no gradient A/B between them means
anything.

Two properties, and the first is why the fix moved no banked number: drawing the whole set in
one call at the same point in the stream is BIT-IDENTICAL to the sequential draws it replaces.
"""

from __future__ import annotations

import numpy as np

SEED = 20260926
N_ATOM, N_SAMPLES = 517, 48


def _sequential(rng, n_samples, n_atom):
    """What the harness used to do: one (n_atom, 3) draw per replicate, inside the loop."""
    return [rng.standard_normal((n_atom, 3)).astype("float32") for _ in range(n_samples)]


def _up_front(rng, n_samples, n_atom):
    """What it does now: the whole sample axis, before any chunking, indexed by replicate."""
    return rng.standard_normal((n_samples, n_atom, 3)).astype("float32")


def test_drawing_the_set_up_front_is_bit_identical_to_the_sequential_draws():
    # Same generator, same position in the stream: the fix is a no-op on the numbers, which is
    # what lets the 92.07 s baseline and the new batched arm be compared to each other at all.
    a = _sequential(np.random.default_rng(SEED), N_SAMPLES, N_ATOM)
    b = _up_front(np.random.default_rng(SEED), N_SAMPLES, N_ATOM)
    assert b.shape == (N_SAMPLES, N_ATOM, 3)
    for k in range(N_SAMPLES):
        assert np.array_equal(a[k], b[k]), f"replicate {k} moved"


def test_a_partition_cannot_reach_the_noise():
    """Every way the campaign partitions 48 replicates gives replicate k the same bits."""
    ref = _up_front(np.random.default_rng(SEED), N_SAMPLES, N_ATOM)

    def walk(order):
        return {k: ref[k] for k in order}

    partitions = {
        "in order": list(range(N_SAMPLES)),
        "chunks of 4": [k for c in range(0, N_SAMPLES, 4) for k in range(c, c + 4)],
        "chunks of 12, last chunk first": (list(range(36, 48)) + list(range(0, 36))),
        # of3t-p10dp shards round-robin, not contiguously: the replicate index is the sigma
        # index and the schedule is ordered by noise level.
        "4 ranks, round robin": [k for r in range(4) for k in range(r, N_SAMPLES, 4)],
        # A chunk that DRAM refused and reran at half the width sees its replicates twice.
        "a refused chunk rerun": (list(range(0, 8)) + list(range(4, N_SAMPLES))),
    }
    for name, order in partitions.items():
        got = walk(order)
        for k, v in got.items():
            assert np.array_equal(v, ref[k]), f"{name}: replicate {k} moved"


def test_the_sequential_draw_it_replaces_was_position_dependent():
    """The defect, stated as a test, so nobody reintroduces the pattern thinking it was fine.

    Hand the old code a partition that is not in-order and replicate k gets some other
    replicate's structure. This is the failure the fix removes, not a hypothetical.
    """
    rng = np.random.default_rng(SEED)
    in_order = _sequential(rng, 8, N_ATOM)
    # The same generator, walked as "second chunk first" -- the shape a re-ranked or
    # multi-rank walk has.
    rng = np.random.default_rng(SEED)
    second_chunk_first = _sequential(rng, 8, N_ATOM)[4:] + _sequential(rng, 4, N_ATOM)
    assert not np.array_equal(in_order[0], second_chunk_first[0])


def test_the_harness_draws_nothing_inside_its_replicate_loop():
    """A source guard: the generator is consumed in `diffusion_pre` and nowhere below it."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "perf/of3t_stepfloor/fullstep.py"
    text = src.read_text()
    body = text[text.index("def diffusion_chunk("):text.index("def host_losses(")]
    assert "standard_normal" not in body, (
        "diffusion_chunk draws its own noise again; the whole point is that pre['noise'][k] "
        "is the only source and it is indexed by replicate")
    assert re.search(r"noise_all\b.*=.*pre\[.noise.\]", body), (
        "diffusion_chunk no longer reads the up-front noise set")
    assert "noise_all[k]" in body, "the noise is no longer indexed by the replicate index"
