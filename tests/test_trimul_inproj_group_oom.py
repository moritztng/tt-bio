"""A trimul in-projection width DRAM has refused must never be offered again.

`_TRIMUL_INPROJ_FUSED_BYTES` caps the fused in-projection at 1 GiB, and that cap is a
footprint budget measured with 6 GiB of foreign DRAM held -- not a fit test against what a
given fold has live. At OpenDDE's 1024-residue shape it names group 4, whose 1 GiB output plus
the four-way split it feeds is this module's whole DRAM peak, and the device refuses the split:
268435456 B wanted against 14732256 B free per bank, byte-identical across two separate runs.

So the width has to come down after the refusal, and it may only come down after one: every
group is bit-exact against every other, but a pre-emptive narrowing would still change the
launch count at every size that folds today. Host-only -- no device, no network.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

GRID = (8, 9)     # Wormhole, the grid the 1024 refusal was measured on
SEQ = 1024        # the residue rung that DRAM refuses
CHUNK = 32        # trimul channel-chunk width on the DRAM path
PAIRS = 12        # OpenDDE's channel loop: trimul hidden 384 / chunk 32
HIDDEN = CHUNK * PAIRS


@pytest.fixture
def shape(monkeypatch):
    """Pin the grid and start from an empty refusal record."""
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", GRID)
    monkeypatch.setattr(T, "_FAST_MODE", False)
    monkeypatch.setattr(T, "_TRIMUL_INPROJ_GROUP_CAP", {})


def _group(seq=SEQ, chunk=CHUNK, batch=1, pairs=PAIRS):
    return T._trimul_inproj_group(seq, chunk, batch, pairs)


def _refuse(group, seq=SEQ, batch=1, hidden=HIDDEN):
    T._record_trimul_inproj_oom(seq, hidden, batch, group)


def test_the_refused_width_is_the_one_the_budget_picks(shape):
    """The negative control. Without it, every assertion below could pass vacuously.

    If the byte budget did not offer 4 here, this shape would never have hit the refusal that
    motivates the retry, and a test that only checked "narrowing narrows" would prove nothing
    about the fold that failed.
    """
    assert _group() == 4
    fused = 4 * CHUNK * SEQ * SEQ * 2
    assert 4 * fused == 1024 * 2 ** 20 == T._TRIMUL_INPROJ_FUSED_BYTES


def test_a_refused_width_is_never_offered_again(shape):
    _refuse(4)
    assert _group() == 3


def test_successive_refusals_walk_the_divisors_down_to_one(shape):
    """Each refusal must strictly narrow, or the retry loop would not terminate."""
    seen = []
    for _ in range(6):
        g = _group()
        seen.append(g)
        if g == 1:
            break
        _refuse(g)
    assert seen == [4, 3, 2, 1]


def test_group_one_is_the_floor(shape):
    """At 1 there is nothing left to give and the op re-raises, so the width must stay 1
    rather than wrap or go negative."""
    for g in (4, 3, 2, 1):
        _refuse(g)
    assert _group() == 1


def test_only_the_refused_shape_narrows(shape):
    """The record is keyed on the shape, so one target's refusal must not narrow another's.

    A batched confidence head and a different trimul latent width are different shapes that
    reach the same function, and the 1024 refusal says nothing about either.
    """
    others = {
        "batch 2": (_group(batch=2), lambda: _group(batch=2)),
        "4-pair loop": (_group(pairs=4), lambda: _group(pairs=4)),
        "512 aa": (_group(seq=512), lambda: _group(seq=512)),
    }
    _refuse(4)
    assert _group() == 3, "the refused shape must narrow, or this test proves nothing"
    for name, (before, again) in others.items():
        assert again() == before, name


def test_the_byte_budget_still_bounds_the_width(shape):
    """Narrowing after a refusal may not step over the cap the budget already enforced."""
    budget = T._TRIMUL_INPROJ_FUSED_BYTES
    for seq in (256, 512, 768, 1024):
        fused = 4 * CHUNK * seq * seq * 2
        _refuse(PAIRS + 1, seq=seq)    # a refusal wider than the loop: no extra constraint
        g = _group(seq=seq)
        assert PAIRS % g == 0, (seq, g)
        assert g * fused <= budget, (seq, g)
