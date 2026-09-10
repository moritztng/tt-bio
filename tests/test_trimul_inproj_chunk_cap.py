"""A refused in-projection width must not be re-probed once per pairformer block.

`_record_trimul_inproj_oom` halves a shape's fused byte budget after the device refuses an
allocation, and `_trimul_inproj_group` reads that budget. The channel-chunk width did not: every
call re-entered the loop at `_trimul_chunk_size`'s tuned width, was refused there again, halved the
budget again, and narrowed to satisfy a budget one halving smaller than the block before it. Over a
fold the budget walks to nothing and every block ends at chunk 1, i.e. `hidden` channel passes.

That is what a 1024-residue OpenDDE fold hit on Wormhole: 2016 structural tokens, DRAM 99 % full,
"refused the fused in-projection at chunk 32 x group 1 (seq 2016): retrying at chunk 1" printed
once per block at 4-5 minutes a block, and the fold ran out its 2400 s timeout instead of failing
(logs/fix1024a.log, 2026-09-08).

The device is modelled here as a fit test against a fixed free-DRAM figure -- the one measured at
the refusal, 5777408 B largest contiguous free per bank across 12 banks. Host-only: no device, no
network.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

GRID = (8, 9)          # Wormhole, the grid the refusal was measured on
TOKENS = 2016          # the refiner's structural-token axis for a 1024-residue fold
HIDDEN = 384           # OpenDDE's trimul latent width
CHUNK = 32             # `TRIANGLE_MULT_CHUNK_SIZE`, the tuned minimum on the DRAM path
FREE = 5777408 * 12    # what the allocator had left at the refusal, measured


@pytest.fixture
def shape(monkeypatch):
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", GRID)
    monkeypatch.setattr(T, "_FAST_MODE", False)
    monkeypatch.setattr(T, "_TRIMUL_INPROJ_FUSED_CAP", {})
    # Wormhole returns from the sub-tile slice the bottom of this ladder asks for; Blackhole
    # wedges on it, and stops at one tile instead (test_trimul_inproj_tile_floor.py).
    monkeypatch.setattr(T, "_SUB_TILE_SLICE_WEDGES", False)


def _fused(chunk, group=1):
    return 4 * chunk * group * TOKENS * TOKENS * 2


def _request(chunk):
    """The allocation the device actually refused: one quarter of the fused output."""
    return _fused(chunk) // 4


def _block(start, old):
    """One pairformer block's in-projection, returning (width used, refusals paid).

    `old` re-enters at the tuned width every call, which is the behaviour under test; the fixed
    path enters at whatever the recorded budget allows.
    """
    chunk = start if old else T._trimul_inproj_chunk_cap(TOKENS, HIDDEN, 1, start)
    refusals = 0
    while _request(chunk) > FREE:
        refusals += 1
        T._record_trimul_inproj_oom(TOKENS, HIDDEN, 1, _fused(chunk))
        narrowed = T._trimul_inproj_chunk_cap(TOKENS, HIDDEN, 1, chunk)
        if narrowed == chunk:
            pytest.fail("nothing left to give: the module re-raises here")
        chunk = narrowed
    return chunk, refusals


def _fold(old, blocks=8):
    return [_block(CHUNK, old) for _ in range(blocks)]


def test_the_cap_is_inert_until_the_device_has_refused_something(shape):
    """The control. A size that folds today must keep its width, launch count and arithmetic.

    Every width is bit-exact against every other, but entering narrower than the tuned width
    would still change the launch count at every size that folds, so this may only bite after a
    refusal has been recorded for the shape.
    """
    for tokens in (1504, 1760, TOKENS):        # the 768, 896 and 1024 residue rungs
        assert T._trimul_inproj_chunk_cap(tokens, HIDDEN, 1, CHUNK) == CHUNK
    assert not T._TRIMUL_INPROJ_FUSED_CAP     # reading it records nothing


def test_the_next_block_starts_where_the_last_one_landed(shape):
    """The fix: two refusals for the whole fold, and every later block enters at the width that
    fits."""
    fold = _fold(old=False)
    assert [w for w, _ in fold] == [8] * 8
    assert [r for _, r in fold] == [2, 0, 0, 0, 0, 0, 0, 0]


def test_re_probing_the_refused_width_walks_the_budget_to_chunk_1(shape):
    """The defect, reproduced: the width collapses to 1 and the refusal is paid every block.

    Chunk 1 is 384 channel passes against 48 at the width that actually fits -- an 8x launch
    count for the same arithmetic, on top of one refused 260 MB allocation per block.
    """
    fold = _fold(old=True)
    assert [w for w, _ in fold] == [8, 4, 2, 1, 1, 1, 1, 1]
    assert [r for _, r in fold] == [2, 1, 1, 1, 1, 1, 1, 1]
    assert HIDDEN // 1 == 8 * (HIDDEN // 8)


def test_the_width_the_fix_settles_on_is_the_widest_that_fits(shape):
    """Not merely narrower: the loop must not overshoot past a width the device would take."""
    _fold(old=False, blocks=2)
    settled = T._trimul_inproj_chunk_cap(TOKENS, HIDDEN, 1, CHUNK)
    assert _request(settled) <= FREE < _request(settled * 2)
