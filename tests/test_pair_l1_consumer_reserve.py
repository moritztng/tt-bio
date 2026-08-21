"""The pair tensor's L1 residency must leave its consumers room, at every token count.

Boltz-2 could not fold anything padding to 704 tokens on a p150a: `PairWeightedAveraging`'s
per-head softmax threw at program creation, because the gate had admitted a 953 KB/core normed
pair tensor and the softmax's static circular buffers needed 563658 B where 543 KB was left.
640 aa plus a 20-atom ligand is 660 tokens, which pads to 704, so a ligand fixture found it and
an apo ladder on the 64-aa lattice did not; 641-704 aa apo dies identically.

The gate priced a multiple of the tensor against the whole grid's L1. Interleaved bytes summed
across banks are not the wall -- the wall is per core, the tensor scales with its area and its
consumers' circular buffers scale with the row width, so no multiple of the tensor is the right
shape for the reserve. `_PAIR_L1_CONSUMER_RESERVE` prices it in bytes per core instead.

Host-only: pure allocator arithmetic, stubbed part figures, no device and no fold.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tt_bio import tenstorrent as T

# p150a, measured: `ttnn.get_max_worker_l1_unreserved_size()` on a 13x10 Blackhole grid.
PER_CORE = 1532448
GRID = (13, 10)
# Measured at the shape that crashed (704 padded tokens): the live L1 buffers held 1230848
# B/core, of which the normed pair tensor and its projection were 983638, and the softmax's
# static circular buffers needed 316448 more. 1230848 - 983638 + 316448.
MEASURED_NEED_AT_704 = 563658
PAIR_CHANNELS = 128
LADDER = [448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088]
# Every rung that folded before the fix, and must fold identically after it.
ADMITTED = {448, 512, 576, 640}


@pytest.fixture
def part(monkeypatch):
    """Pin the p150a figures so the arithmetic is the test, not the host."""
    monkeypatch.setattr(T.ttnn, "get_max_worker_l1_unreserved_size", lambda: PER_CORE)
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", GRID)


def pair(tokens: int, channels: int = PAIR_CHANNELS):
    """A normed pair tensor as the two call sites hand it to the gate."""
    return SimpleNamespace(shape=(tokens, tokens, channels), dtype=T.ttnn.bfloat16)


def admits(tokens: int, channels: int = PAIR_CHANNELS) -> bool:
    mc = T._l1_memory_config_if_it_fits(pair(tokens, channels), 1.0,
                                        T._PAIR_L1_CONSUMER_RESERVE)
    return mc is T.ttnn.L1_MEMORY_CONFIG


def test_the_704_token_rung_is_refused(part):
    """The crash. 640 aa + a 20-atom ligand, and 641-704 aa apo, both land here."""
    assert not admits(704)


def test_the_old_gate_admitted_it(part):
    """Without which this test would pass on the broken build too."""
    assert T._l1_memory_config_if_it_fits(pair(704), 1.5) is T.ttnn.L1_MEMORY_CONFIG


def test_rungs_that_folded_before_still_go_to_l1(part):
    """Bit-exactness: the fix may only change the rung that crashed.

    Verified on hardware as well -- 256/512/576/704/768 aa with the ligand fixture give
    byte-identical structures across the fix.
    """
    for tokens in LADDER:
        assert admits(tokens) is (tokens in ADMITTED), tokens


def test_the_decision_is_monotone_in_token_count(part):
    """Why this is not a patch at one size: no second window can open at 896 or 1024.

    The condition is area against a fixed budget, so it is monotone -- once the gate says DRAM
    it never says L1 again at any larger shape.
    """
    seen_dram = False
    for tokens in LADDER:
        if not admits(tokens):
            seen_dram = True
        else:
            assert not seen_dram, f"L1 came back at {tokens} after a refusal"


def test_the_reserve_covers_what_the_consumers_measured(part):
    """A reserve smaller than the need is the bug with a new constant."""
    assert T._PAIR_L1_CONSUMER_RESERVE >= MEASURED_NEED_AT_704


def test_the_reserve_still_fits_the_512_aa_production_shape(part):
    """The other end of the window: too large a reserve costs the shipped fold its lever."""
    cores = GRID[0] * GRID[1]
    keeps_576 = PER_CORE - 576 * 576 * PAIR_CHANNELS * 2 // cores
    refuses_704 = PER_CORE - 704 * 704 * PAIR_CHANNELS * 2 // cores
    assert refuses_704 < T._PAIR_L1_CONSUMER_RESERVE <= keeps_576


def test_a_part_with_less_l1_per_core_only_gets_more_conservative(monkeypatch):
    """An absolute per-core reserve scales the safe way onto a part we do not own."""
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", (11, 10))
    for per_core in (PER_CORE, PER_CORE // 2, T._PAIR_L1_CONSUMER_RESERVE - 1):
        monkeypatch.setattr(T.ttnn, "get_max_worker_l1_unreserved_size", lambda p=per_core: p)
        assert not admits(704)


def test_both_pair_sites_pass_the_reserve():
    """One site fixed and one forgotten is the same crash from the other module."""
    src = (Path(T.__file__)).read_text()
    assert src.count("_l1_layer_norm(z, 1.0, _PAIR_L1_CONSUMER_RESERVE") == 2
    assert "_l1_layer_norm(z, 1.5" not in src
