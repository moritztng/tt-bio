"""The footprint cap may not buy its footprint with a sub-tile slice.

`_trimul_inproj_chunk_cap` narrows the trimul's channel chunk until the fused in-projection fits
a byte budget, and the channel loop consumes that projection with a 4-way `ttnn.chunk` along the
last axis -- so the chunk width IS the slice width. Below one tile that slice stops being a
narrower version of the same op: measured on a p150a at the fold's own shape, 4 x 32-wide pieces
out of `[1, 2208, 2208, 128]` return in 27.982 ms while 4 x 16-wide pieces out of
`[1, 2208, 2208, 64]` never return at all (perf/bgsdpa/repro_chunk_fix.json).

`4 * 32 * seq^2 * 2` crosses the 1 GiB default at exactly seq 2048, so before the floor every
trimul-using model on the DRAM path was narrowed to a half tile the moment its padded token count
passed 2048. That is what BoltzGen's "capacity ceiling above 14786 atoms" was: 1831 residues pads
to 1856 and folds, 2100 pads to 2208 and hung.

The floor is on the DEFAULT budget only. A recorded budget is an allocation the device actually
refused, and OpenDDE's 1024-residue Wormhole fold is verified 3/3 byte-identical at chunk 8, so
that ladder still walks all the way down (`test_trimul_inproj_chunk_cap.py` covers it).
Host-only: no device, no network.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

TILE = 32
# (cz, hidden) of the models that reach the DRAM trimul path.
MODELS = [("boltzgen", 64, 32), ("boltz2", 128, 128), ("protenix-v2", 128, 128),
          ("openfold3", 128, 128), ("opendde", 384, 384), ("esmfold2", 128, 128)]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(T, "_TRIMUL_INPROJ_FUSED_CAP", {})
    monkeypatch.setattr(T, "_IS_SMALL_GRID", False)


def _width(seq, hidden):
    chunk = T._trimul_chunk_size(seq, hidden, 1)
    return T._trimul_inproj_chunk_cap(seq, hidden, 1, chunk)


def test_the_footprint_cap_never_asks_for_a_sub_tile_slice():
    """The defect. Every one of these sizes was halved to 16 or 8 with nothing refused."""
    for name, _cz, hidden in MODELS:
        for seq in (2080, 2208, 2304, 3072, 4096):
            w = _width(seq, hidden)
            assert w % TILE == 0, f"{name} at seq {seq} would slice {w} wide"


def test_2048_is_the_boundary_and_the_shipped_width_is_unchanged_below_it():
    """The control: the floor may not move a size that already folds.

    `4 * 32 * seq^2 * 2 <= 1 GiB` iff seq <= 2048, so 2048 is the last size the cap left alone
    and 2080 the first it halved. Everything at or below it must keep the tuned width, because a
    narrower chunk is a different launch count for the same arithmetic.
    """
    for _name, _cz, hidden in MODELS:
        for seq in (1024, 1504, 1760, 1856, 2016, 2048):
            assert _width(seq, hidden) == T.TRIANGLE_MULT_CHUNK_SIZE
    assert 4 * TILE * 2048 ** 2 * 2 <= T._TRIMUL_INPROJ_FUSED_BYTES
    assert 4 * TILE * 2080 ** 2 * 2 > T._TRIMUL_INPROJ_FUSED_BYTES


def test_a_refusal_may_go_below_a_tile_only_where_that_slice_returns(monkeypatch):
    """Only the footprint preference has an unconditional floor.

    A recorded budget is an allocation the device actually refused, so on Wormhole -- where a
    sub-tile slice comes back, and where OpenDDE's 1024-residue fold is verified 3/3
    byte-identical at chunk 8 -- narrowing past a tile is better than not folding. On Blackhole
    the same slice never returns, so there is nothing below a tile to narrow to and the caller
    re-raises the refusal instead.
    """
    seq, hidden = 2016, 384
    fused = 4 * TILE * seq * seq * 2
    for wedges, expect_below in ((False, True), (True, False)):
        monkeypatch.setattr(T, "_SUB_TILE_SLICE_WEDGES", wedges)
        monkeypatch.setattr(T, "_TRIMUL_INPROJ_FUSED_CAP", {})
        assert _width(seq, hidden) == TILE                 # nothing refused yet: same width
        T._record_trimul_inproj_oom(seq, hidden, 1, fused)
        got = T._trimul_inproj_chunk_cap(seq, hidden, 1, TILE)
        assert (got < TILE) is expect_below, (wedges, got)


def test_the_part_default_is_the_wedging_side():
    """An unmeasured part gets the wider width, not a spin no timeout catches."""
    assert T._SUB_TILE_SLICE_WEDGES is True


def test_the_group_never_puts_the_slice_back_below_a_tile():
    """The slice width is chunk * group, so the group may only ever widen it."""
    for _name, _cz, hidden in MODELS:
        for seq in (1024, 2048, 2208, 3072):
            chunk = _width(seq, hidden)
            group = T._trimul_inproj_group(seq, chunk, 1, hidden // chunk)
            assert group >= 1 and (chunk * group) % TILE == 0
