"""A trimul in-projection footprint DRAM has refused must come down, and stay down.

`_TRIMUL_INPROJ_FUSED_BYTES` caps the fused in-projection at 1 GiB, and that cap is a footprint
budget measured with 6 GiB of foreign DRAM held -- not a fit test against what a given fold has
live. OpenDDE at 1024 residues refines on a 2016-token structural axis, where the fused output is
0.97 GiB at the NARROWEST group and the four-way split it feeds asks for another 0.97 GiB while it
is still resident. The device refuses: 260112384 B wanted against 15589856 B free per bank.

So the budget has to come down after a refusal, and it has to come down in BYTES. Recording a
refused group width is not enough -- a narrower channel chunk shrinks the fused output and the
group search then wins the same peak straight back as a wider group.

It may only come down after a refusal, though: every (chunk, group) pair is bit-exact against
every other, but narrowing pre-emptively would still change the launch count at every size that
folds today. Host-only -- no device, no network.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

GRID = (8, 9)     # Wormhole, the grid the refusal was measured on
TOKENS = 2016     # the refiner's structural-token axis for a 1024-residue fold
HIDDEN = 384      # OpenDDE's trimul latent width
CHUNK = 32        # `TRIANGLE_MULT_CHUNK_SIZE`, the tuned minimum on the DRAM path


@pytest.fixture
def shape(monkeypatch):
    """Pin the grid and start from an empty refusal record."""
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", GRID)
    monkeypatch.setattr(T, "_FAST_MODE", False)
    monkeypatch.setattr(T, "_TRIMUL_INPROJ_FUSED_CAP", {})


def _fused(chunk, group, tokens=TOKENS, batch=1):
    return 4 * chunk * group * tokens * tokens * batch * 2


def _narrow(chunk, group, tokens=TOKENS, batch=1, hidden=HIDDEN):
    """One retry step, exactly as `TriangleMultiplication.__call__` performs it."""
    T._record_trimul_inproj_oom(tokens, hidden, batch, _fused(chunk, group, tokens, batch))
    budget = T._trimul_inproj_budget(tokens, hidden, batch)
    while (chunk > 1 and hidden % (chunk // 2) == 0
           and _fused(chunk, 1, tokens, batch) > budget):
        chunk //= 2
    return chunk, T._trimul_inproj_group(tokens, chunk, batch, hidden // chunk)


def test_the_group_lever_is_already_exhausted_at_the_refused_shape(shape):
    """The negative control, and the reason a group-only retry did not fix 1024.

    If the shipped budget offered any group above 1 here, narrowing the group would have been
    the whole answer and the chunk lever below would be untested dead weight.
    """
    assert T._trimul_inproj_group(TOKENS, CHUNK, 1, HIDDEN // CHUNK) == 1
    assert _fused(CHUNK, 1) == 1040449536 <= T._TRIMUL_INPROJ_FUSED_BYTES
    assert _fused(CHUNK, 1) // 4 == 260112384          # the request the device refused


def test_a_refusal_narrows_the_channel_chunk(shape):
    assert _narrow(CHUNK, 1) == (16, 1)


def test_successive_refusals_strictly_shrink_the_peak_and_terminate(shape):
    chunk, group = CHUNK, 1
    peaks = [_fused(chunk, group)]
    for _ in range(12):
        nxt = _narrow(chunk, group)
        if nxt == (chunk, group):
            break                                       # the op re-raises here
        chunk, group = nxt
        peaks.append(_fused(chunk, group))
    assert peaks == sorted(peaks, reverse=True), peaks
    assert len(set(peaks)) == len(peaks), peaks         # strictly, so the loop cannot spin
    assert chunk == 1


def test_the_budget_never_rises_again(shape):
    """A later, smaller refusal must not raise a cap an earlier one lowered."""
    _narrow(CHUNK, 1)
    low = T._trimul_inproj_budget(TOKENS, HIDDEN, 1)
    _narrow(1, 1)                                       # a tiny refusal at the same shape
    assert T._trimul_inproj_budget(TOKENS, HIDDEN, 1) <= low


def test_only_the_refused_shape_narrows(shape):
    """The record is keyed on the shape, so one target's refusal must not narrow another's."""
    others = {
        "batch 2": lambda: T._trimul_inproj_group(TOKENS, CHUNK, 2, HIDDEN // CHUNK),
        "128-wide latent": lambda: T._trimul_inproj_group(TOKENS, CHUNK, 1, 4),
        "512 tokens": lambda: T._trimul_inproj_group(512, CHUNK, 1, HIDDEN // CHUNK),
    }
    before = {k: f() for k, f in others.items()}
    _narrow(CHUNK, 1)
    assert T._trimul_inproj_budget(TOKENS, HIDDEN, 1) < T._TRIMUL_INPROJ_FUSED_BYTES, \
        "the refused shape must narrow, or this test proves nothing"
    for k, f in others.items():
        assert f() == before[k], k


def test_an_empty_record_returns_exactly_the_shipped_width(shape):
    """Nothing that folds today may change, and here that is an identity, not a measurement.

    With no refusal recorded the budget is `_TRIMUL_INPROJ_FUSED_BYTES`, so every shape keeps
    its width, its launch count and its arithmetic until the device itself refuses one. The
    oracle below is the pre-fix expression written out independently.
    """
    for tokens in (128, 256, 384, 512, 768, 1024, 2016, 2048):
        for pairs, chunk in ((4, 32), (12, 32), (4, 64)):
            for batch in (1, 2, 5):
                fused = _fused(chunk, 1, tokens, batch)
                want = 1
                for g in range(min(pairs, T._TRIMUL_INPROJ_GROUP), 1, -1):
                    if pairs % g == 0 and g * fused <= T._TRIMUL_INPROJ_FUSED_BYTES:
                        want = g
                        break
                got = T._trimul_inproj_group(tokens, chunk, batch, pairs)
                assert got == want, (tokens, chunk, batch, pairs, got, want)
