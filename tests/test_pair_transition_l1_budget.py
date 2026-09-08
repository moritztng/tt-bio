"""Which pair-transition row blocks get L1, on each part class.

No device: the decision is arithmetic over the part's L1 total and the block's shape, and it is
pinned here for both part classes at once, on any part, for the same reason
`_concat_host_budget` is. The two numbers that matter are the ones the parts report --
100470528 B on a 12 GiB Wormhole Galaxy chip (72 banks x 1395424 B) and 190028800 B on a p150a
(130 banks x 1461760 B) -- so a budget that is right on one of them and dead on the other shows
up as a differing verdict here rather than as a throw on a card nobody was measuring.
"""
import pytest

from tt_bio.rfd3.model import (_PAIR_TRANSITION_H_CHUNK, _pair_transition_chunk_h,
                               _pair_transition_l1_fits)
from tt_bio.tenstorrent import L1_TOTAL_BYTES_WORMHOLE, l1_resident_budget_bytes

WORMHOLE = L1_TOTAL_BYTES_WORMHOLE      # 100470528
P150A = 190_028_800
Z_HIDDEN = 512                          # z_transition: c_z=128, n=4 -- the site that threw
PAIRFORMER_HIDDEN = 256                 # transition_1.{0,1}: c_z=128, n=2


@pytest.fixture
def budget(monkeypatch):
    def use(n):
        monkeypatch.setenv("TT_BIO_L1_RESIDENT_BUDGET_BYTES", str(n))
        assert l1_resident_budget_bytes() == n
    return use


def _fits(tokens, hidden, budget_bytes, residents=2):
    h = _pair_transition_chunk_h(tokens, hidden, tokens)
    return _pair_transition_l1_fits(tokens, hidden, h, residents), h


@pytest.mark.parametrize("tokens", [512, 544, 640, 704])
def test_wormhole_keeps_the_residency_up_to_704(budget, tokens):
    # Every size the 704 ceiling was measured at keeps the L1-resident path it was measured on.
    budget(WORMHOLE)
    fits, h = _fits(tokens, Z_HIDDEN, WORMHOLE)
    assert h == _PAIR_TRANSITION_H_CHUNK
    assert fits


@pytest.mark.parametrize("tokens", [768, 832, 896, 960, 992, 1024])
def test_wormhole_declines_from_768_up(budget, tokens):
    # 768 is where the shipped code asked for two 700416 B-per-bank buffers out of a 1395424 B
    # bank and missed by 5408 B. The budget declines the residency instead, at 768 and at every
    # size above it, and 704 above still keeps it: the negative control for this gate is that
    # it does NOT decline everything.
    budget(WORMHOLE)
    fits, h = _fits(tokens, Z_HIDDEN, WORMHOLE)
    assert h == _PAIR_TRANSITION_H_CHUNK
    assert not fits


@pytest.mark.parametrize("tokens", [512, 640, 704, 768, 832, 896, 960, 1024])
def test_a_p150a_keeps_the_residency_at_every_size(budget, tokens):
    # 1.9x the L1, so nothing in range declines and Blackhole never changes path. A gate that
    # declined here would be a perf regression on the part that does not need it.
    budget(P150A)
    fits, _ = _fits(tokens, Z_HIDDEN, P150A)
    assert fits


@pytest.mark.parametrize("tokens", [512, 704, 768, 1024])
def test_the_narrower_pairformer_transition_never_declines(budget, tokens):
    # hidden=256 halves the residents, so the sites that did not throw do not change either.
    budget(WORMHOLE)
    fits, _ = _fits(tokens, PAIRFORMER_HIDDEN, WORMHOLE)
    assert fits


def test_the_shipped_constant_would_have_declined_nothing():
    # Why the gate could not be `_pair_transition_chunk_h`'s own budget: 138000000 B is 1.37x
    # the L1 a Wormhole part has, so priced against it the 768-token block "fits" and the
    # allocator is asked for a buffer that cannot exist.
    from tt_bio.rfd3.model import _PAIR_TRANSITION_L1_BYTES
    assert _PAIR_TRANSITION_L1_BYTES > WORMHOLE
    h = _pair_transition_chunk_h(768, Z_HIDDEN, 768)
    assert h == _PAIR_TRANSITION_H_CHUNK
    assert 2 * 2 * 768 * Z_HIDDEN * h > WORMHOLE


def test_zero_means_no_budget(monkeypatch):
    # The A/B baseline and the way back at a release gate: every residency granted, which is
    # the unconditional-L1 path the code shipped with.
    monkeypatch.setenv("TT_BIO_L1_RESIDENT_BUDGET_BYTES", "0")
    fits, _ = _fits(1024, Z_HIDDEN, 0)
    assert fits


def test_the_gate_cannot_move_the_chunk_height(budget):
    # The height is arithmetic and the destination is not, so this is the property the whole
    # design rests on: the same (tokens, hidden) gives the same h on both parts.
    for tokens in (512, 640, 704, 768, 832, 896, 960, 1024):
        for hidden in (Z_HIDDEN, PAIRFORMER_HIDDEN):
            budget(WORMHOLE)
            wh = _pair_transition_chunk_h(tokens, hidden, tokens)
            budget(P150A)
            assert _pair_transition_chunk_h(tokens, hidden, tokens) == wh
