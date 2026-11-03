"""A trimul narrowed below a tile may not pay for the narrowing in padded DRAM.

Seed 4101 of WH 1024 Mol-1 (mol1-fair-tt, main b73730028, 8j2w at 1000 tokens) refused trimul's
fused in-projection in a heavy MSA mode and the ladder walked chunk 32 -> 16 -> 8 -> 4 -> 2 -> 1,
each step refused the same 64 MiB: the projection is TILE laid out, so below chunk 8 its last axis
(4 x chunk) is one padded tile whatever the chunk. The recorded cap then started every later trimul
of that shape at chunk 1, whose output slices were joined on the device at a whole tile of lanes
each, and the lightest mode died there at 12.7 of 12.9 GB with 5.1 GiB in use at the call's start
(state/mol1/evidence/wh1024oom/ROOT.md). Host-only: no device.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

TILE = 32


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(T, "_TRIMUL_INPROJ_FUSED_CAP", {})
    monkeypatch.setattr(T, "_IS_SMALL_GRID", True)
    monkeypatch.setattr(T, "_SUB_TILE_SLICE_WEDGES", False)   # Wormhole: a refusal may go below a tile


def _refuse_until_settled(seq, hidden, chunk):
    """The handler's loop: record the refused fused bytes, narrow, until a refusal changes nothing."""
    seen = [chunk]
    while True:
        T._record_trimul_inproj_oom(seq, hidden, 1, 4 * chunk * seq * seq * 2)
        nxt = T._trimul_inproj_chunk_cap(seq, hidden, 1, chunk)
        if nxt == chunk:
            return seen
        chunk = nxt
        seen.append(chunk)


@pytest.mark.parametrize("hidden", [128, 384, 448])
def test_the_ladder_stops_where_the_padded_projection_stops_shrinking(hidden):
    seen = _refuse_until_settled(1024, hidden, TILE)
    assert seen == [32, 16, 8], seen
    pad = lambda c: -(-4 * c // TILE) * TILE  # noqa: E731
    assert all(pad(b) < pad(a) for a, b in zip(seen, seen[1:]))


def test_opendde_still_reaches_chunk_8():
    """OpenDDE's 1024-residue Wormhole fold (2016 tokens) folds byte-identical at chunk 8."""
    assert _refuse_until_settled(2016, 384, TILE)[-1] == 8


def test_widths_at_or_above_a_tile_are_unchanged():
    """Above the floor the padded and logical bytes agree, so no shipped width moves."""
    for seq in (1024, 2048, 2208, 3072):
        for hidden in (64, 128, 384):
            c = T._trimul_chunk_size(seq, hidden, 1)
            assert T._trimul_inproj_chunk_cap(seq, hidden, 1, c) == c or seq > 2048


class _X:
    def __init__(self, dtype):
        self.dtype = dtype


def test_sub_tile_slices_join_on_the_host(monkeypatch):
    monkeypatch.setattr(T, "_dtype", lambda: T.ttnn.bfloat16)
    bf16 = _X(T.ttnn.bfloat16)
    assert [T._sub_tile_join(bf16, c) for c in (8, 16, 32, 64)] == [True, True, False, False]
    # bf8 and fp32 blocks would round-trip through torch bf16 lossily: those keep the device join.
    assert not T._sub_tile_join(_X(T.ttnn.float32), 8)


def test_a_lighter_restart_starts_trimul_at_the_tuned_width():
    """The refused ladder's budget belongs to the DRAM that refused it, not to the lighter mode
    the trunk restarts in after freeing that DRAM: 4101's park restart had 7 GiB free."""
    _refuse_until_settled(1024, 448, TILE)
    assert T._trimul_inproj_chunk_cap(1024, 448, 1, TILE) < TILE
    T.forget_trimul_inproj_refusals()
    assert T._trimul_inproj_chunk_cap(1024, 448, 1, TILE) == TILE


def test_the_trunk_step_down_forgets_the_refusals():
    import inspect
    import tt_bio.protenix as P
    assert "forget_trimul_inproj_refusals()" in inspect.getsource(P.Trunk.__call__)
