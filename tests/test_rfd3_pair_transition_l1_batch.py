"""The pair Transition's L1 chunk budget has to know about the batch.

`_PAIR_TRANSITION_L1` keeps `fc2`'s output and the gated product resident, and sizes the
chunk against a measured-safe 138 MB of live L1. The size formula counted the two residents
and the pair width but not the batch, so at b=2 it asked for twice the L1 it had budgeted
and the second allocation failed -- the OOM that closed RFD3 batching three times
(`perf/p76/batch_r4_qb2.log`, request 92 274 688 B against 68.4 MB free).

Host-only: the formula is arithmetic, so the regression is testable without a card.
"""
from __future__ import annotations

import pytest

from tt_bio.rfd3.model import (
    _PAIR_TRANSITION_H_CHUNK,
    _PAIR_TRANSITION_L1_BYTES,
    _pair_transition_chunk_h,
)

# The page fixture's token pair: 685 rows padded to 704 wide, z_transition at hidden=512.
H, W_PAD, HIDDEN = 685, 704, 512
L1_TOTAL_BYTES = 110 * 1_461_760          # p300c: 110 banks, 1 461 760 B each


def live_bytes(batch, h):
    """`b` + `m`, both [batch, h, W_PAD, HIDDEN] bf16."""
    return 2 * batch * h * W_PAD * HIDDEN * 2


def test_b1_is_unchanged():
    """The fix must not move the shipped default off its measured optimum."""
    assert _pair_transition_chunk_h(1, W_PAD, HIDDEN, H) == _PAIR_TRANSITION_H_CHUNK == 64


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_live_footprint_stays_inside_the_budget(batch):
    h = _pair_transition_chunk_h(batch, W_PAD, HIDDEN, H)
    assert h >= 1
    assert live_bytes(batch, h) <= _PAIR_TRANSITION_L1_BYTES


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_the_b1_chunk_would_not_have_fitted(batch):
    """Guards the direction of the fix, not just its bound: the batch-blind h is what the
    allocator rejected, so a regression that drops the batch term is caught here rather than
    on a card."""
    assert live_bytes(batch, _PAIR_TRANSITION_H_CHUNK) > L1_TOTAL_BYTES
    assert live_bytes(batch, _pair_transition_chunk_h(batch, W_PAD, HIDDEN, H)) < L1_TOTAL_BYTES


def test_the_exact_b2_request_that_crashed():
    """One resident at the batch-blind h=64, b=2 -- the byte figure in the p76 log."""
    assert live_bytes(2, _PAIR_TRANSITION_H_CHUNK) // 2 == 92_274_688
