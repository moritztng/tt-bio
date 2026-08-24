"""What batch a design request of an L-atom target actually gets.

`effective_design_batch` is the whole of RFD3's batching policy: four bounds, tightest wins. It
decides, among other things, that asking for 8 designs of a 6051-atom target runs them one at a
time. That is a user-visible default, and until now it was an inline expression in `run_design`
with nothing pinning it, so a tree move could change it silently and the only signal would have
been a published second-per-design number moving.

The sizes here are the five RFD3 perf fixtures (`perf/dsfix/fixtures/rfd3_R[0-4].json`), so the
table below is the same ladder the perf work measures. Host-only: it is arithmetic.
"""
from __future__ import annotations

import pytest

from tt_bio.rfd3.design import (
    _BATCH_ATOM_PAIR_BUDGET,
    _BATCH_DESIGN_CEILING,
    _BATCH_SPEED_CAP,
    _BATCH_SPEED_CAP_ABOVE_ATOMS,
    effective_design_batch,
)

# L, what the atom-pair budget alone admits, what a request for 8 designs actually gets.
# R0 2299, R1 2952, R2 3844, R3 4558, R4 6051.
LADDER = [(2299, 17, 8), (2952, 10, 8), (3844, 6, 1), (4558, 4, 1), (6051, 2, 1)]


@pytest.mark.parametrize("L,budget,got", LADDER)
def test_the_batch_a_request_for_eight_designs_gets(L, budget, got):
    assert effective_design_batch(8, L) == got


@pytest.mark.parametrize("L,budget,got", LADDER)
def test_the_atom_pair_budget_term(L, budget, got):
    """Isolated, so a change in the budget constant is attributable and not masked by the cap."""
    assert max(1, _BATCH_ATOM_PAIR_BUDGET // (L * L)) == budget


def test_the_speed_cap_is_what_costs_the_three_large_sizes_their_batch():
    """The cap, not the allocator, is why a large target designs one at a time: the budget would
    have admitted 6, 4 and 2. This is the fact the p116 ladder re-measures."""
    for L, budget, got in LADDER:
        if L > _BATCH_SPEED_CAP_ABOVE_ATOMS:
            assert got == _BATCH_SPEED_CAP < budget


@pytest.mark.parametrize("L,budget,got", LADDER)
@pytest.mark.parametrize("ask", [1, 2, 4, 8, 512, 1024])
def test_it_only_ever_shrinks(L, budget, got, ask):
    """The clamp never hands back more designs than were asked for, so it cannot OOM by itself
    and it cannot turn a 1-design request into a batch."""
    eff = effective_design_batch(ask, L)
    assert 1 <= eff <= ask


@pytest.mark.parametrize("L,budget,got", LADDER)
def test_asking_for_one_always_gets_one(L, budget, got):
    assert effective_design_batch(1, L) == 1


def test_the_budget_falls_as_the_target_grows():
    """It is 1/L**2, so the ladder has to be non-increasing. Guards a swapped bound."""
    admits = [max(1, _BATCH_ATOM_PAIR_BUDGET // (L * L)) for L, _, _ in LADDER]
    assert admits == sorted(admits, reverse=True)


def test_the_design_ceiling_binds_when_nothing_else_does():
    """A tiny target with a huge request: only the ceiling is left to stop it."""
    assert effective_design_batch(10_000, 32) == _BATCH_DESIGN_CEILING == 512


def test_the_cap_does_not_bind_at_or_below_its_threshold():
    """The boundary is exclusive -- `L > _BATCH_SPEED_CAP_ABOVE_ATOMS` -- and 2952 is the largest
    size where batching was measured to still pay, so it must be on the paying side of it."""
    L = _BATCH_SPEED_CAP_ABOVE_ATOMS
    assert effective_design_batch(8, L) == 8
    assert effective_design_batch(8, L + 1) == _BATCH_SPEED_CAP
