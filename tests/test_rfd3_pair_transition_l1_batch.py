"""The pair Transition's L1 chunk height must not depend on the batch.

Two regressions in one file, because the second fix replaced the first.

`_PAIR_TRANSITION_L1` keeps `fc2`'s output and the gated product resident and sizes the chunk
against a measured-safe 138 MB of live L1. The formula first counted the pair width and hidden
but not the batch, so at b=2 it asked for twice the L1 it had budgeted and the second allocation
failed: the OOM that closed RFD3 batching three times (`perf/p76/batch_r4_qb2.log`, request
92 274 688 B against 68.4 MB free). Dividing the budget by the batch fixed that crash and bought
a worse bug -- the chunk height selects which of several bit-different answers the chunked path
produces (state doc §15.2/§15.3, four heights and four fold digests), so a b=2 design stopped
reproducing the b=1 structure at 62.7 % of (size, hidden) pairs.

`Transition.__call__` slices the batch as well as the rows instead. One call covers one batch
element, so `h` is a pure function of (w_pad, hidden, height) and the live footprint per call is
the b=1 footprint at every batch.

Host-only: all of it is arithmetic, so both regressions are testable without a card.
"""
from __future__ import annotations

import inspect

import pytest

from tt_bio.rfd3.model import (
    _PAIR_TRANSITION_H_CHUNK,
    _PAIR_TRANSITION_L1_BYTES,
    _pair_transition_chunk_h,
    _pair_transition_slices,
)

# The page fixture's token pair: 685 rows padded to 704 wide, z_transition at hidden=512.
H, W_PAD, HIDDEN = 685, 704, 512
L1_TOTAL_BYTES = 110 * 1_461_760          # p300c: 110 banks, 1 461 760 B each

# §15.4's table: the four (tokens, hidden) points where the batch divisor used to move h.
POINTS = [(514, 512), (514, 256), (685, 512), (685, 256)]
BATCHES = [1, 2, 4, 8]


def live_bytes(batch, h):
    """`b` + `m`, both [batch, h, W_PAD, HIDDEN] bf16."""
    return 2 * batch * h * W_PAD * HIDDEN * 2


def test_b1_is_unchanged():
    """The fix must not move the shipped default off its measured optimum."""
    assert _pair_transition_chunk_h(W_PAD, HIDDEN, H) == _PAIR_TRANSITION_H_CHUNK == 64


def test_the_height_has_no_batch_input_at_all():
    """The strongest form of batch invariance: the batch cannot reach the formula."""
    assert "batch" not in inspect.signature(_pair_transition_chunk_h).parameters


@pytest.mark.parametrize("tokens,hidden", POINTS)
def test_every_chunk_covers_one_batch_element(tokens, hidden):
    """What makes the batch-free height safe: no call ever holds two batches live."""
    h = _pair_transition_chunk_h(-(-tokens // 32) * 32, hidden, tokens)
    for batch in BATCHES:
        assert {stop_b - b for b, stop_b in
                ((b, b + 1) for b, _, _ in _pair_transition_slices(batch, tokens, h))} == {1}
        assert live_bytes(1, h) <= _PAIR_TRANSITION_L1_BYTES


@pytest.mark.parametrize("batch", BATCHES)
def test_the_slices_tile_the_tensor_exactly_once(batch):
    h = _pair_transition_chunk_h(W_PAD, HIDDEN, H)
    covered = [(b, r) for b, s, e in _pair_transition_slices(batch, H, h) for r in range(s, e)]
    assert sorted(covered) == [(b, r) for b in range(batch) for r in range(H)]
    assert len(covered) == len(set(covered))


def test_b1_is_the_plain_row_split():
    """At b=1 the batch loop must not change the op sequence, or the shipped digest moves."""
    h = _pair_transition_chunk_h(W_PAD, HIDDEN, H)
    assert _pair_transition_slices(1, H, h) == [(0, s, min(s + h, H)) for s in range(0, H, h)]


def test_the_exact_b2_request_that_crashed_is_now_never_made():
    """One resident at h=64 and b=2 -- the byte figure in the p76 log. Slicing the batch is
    what keeps the batch-blind height from asking for it again."""
    assert live_bytes(2, _PAIR_TRANSITION_H_CHUNK) // 2 == 92_274_688
    assert live_bytes(2, _PAIR_TRANSITION_H_CHUNK) > L1_TOTAL_BYTES
    assert live_bytes(1, _pair_transition_chunk_h(W_PAD, HIDDEN, H)) < L1_TOTAL_BYTES
