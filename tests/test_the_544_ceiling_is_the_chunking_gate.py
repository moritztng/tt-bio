"""OpenDDE's published 544-residue Wormhole ceiling is `SEQ_LEN_MORE_CHUNKING` in residues.

`_apply_grid_thresholds` scales that gate by per-core L1 to 640 and then RAISES it from DRAM
capacity: `1024 * sqrt((dram/3) / 3.461 GiB)`, snapped to 32. On a 12 GiB Galaxy Wormhole chip that
is 1088 pair tokens. OpenDDE refines on a structural-token axis 1.945x its residue count
(`test_the_pair_axis_is_not_the_residue_axis`), so:

    544 residues -> 1057 tokens -> 1088 padded == the gate, so the WHOLE-tensor path
    576 residues -> 1120 tokens -> 1120 padded  > the gate, so the CHUNKED path

`size_limits.CEILINGS` records `pass_at=544, fail_at=576` for both checkpoints, measured through
the live API on two independent ladders a day apart. That boundary is not a capacity limit that
happens to land between two rungs: it is the exact token where this gate changes which path the
pair track takes, and two independently measured ladders landed on it.

Whether the switch CAUSES the throw is not settled here -- 576 folds on the chunked path since the
byte-priced reserve and row cap (Pass 3) -- but the coincidence is exact, so anything that moves the
gate moves the wall, and this pins them together. The formula's own comment concedes the point:
"Whether the chunked path this part takes above 608 is actually faster than the unchunked one has
never been measured", which is what `TT_BIO_SEQ_LEN_MORE_CHUNKING` exists to screen.

Host-only: no device, no network. The DRAM total and per-core L1 are stubbed, which is the only way
to evaluate a part this test does not run on.
"""
from __future__ import annotations

from unittest import mock

import pytest

from tt_bio import tenstorrent as T
from tt_bio import size_limits
from tt_bio.token_axis import TOKEN_BUCKET, bucketed_width

GIB = 2 ** 30
WH_GRID = (8, 9)          # the Galaxy chip this task's wall was measured on
WH_DRAM = 12 * GIB        # 12 banks x ~1 GiB
BH_DRAM = 32 * GIB        # p150a

# Structural tokens per rung, MEASURED host-only with the real feature builder on 2026-09-08
# (a backbone token plus a sidechain token per non-GLY residue). Pinned in
# tests/test_the_pair_axis_is_not_the_residue_axis.
TOKENS = {512: 995, 544: 1057, 576: 1120, 608: 1183, 640: 1243,
          768: 1494, 896: 1744, 1024: 1993}

# Every global `_apply_grid_thresholds` writes, so a test cannot leak a retuned constant.
_WRITES = ("_IS_SMALL_GRID", "SEQ_LEN_MORE_CHUNKING", "TRANSITION_BATCH_CHUNKING_THRESHOLD",
           "TRANSITION_W_CHUNKING_THRESHOLD", "TRIANGLE_ATT_CHUNK_SIZE_FAST",
           "TRANSITION_W_CHUNK_SIZE", "TRIANGLE_MULT_L1_MAX_SEQ_FAST", "SMALL_GRID_SEQ_TILE",
           "SMALL_GRID_PAIR_TILE_AREA", "SMALL_GRID_MSA_TILE_AREA", "TRIANGLE_MULT_L1_MAX_SEQ",
           "TRANSITION_L1_CHUNK_BYTES_PER_CORE")


@pytest.fixture
def restore_globals():
    saved = {n: getattr(T, n) for n in _WRITES}
    yield
    for n, v in saved.items():
        setattr(T, n, v)


def _gate(dram, grid=WH_GRID):
    """`SEQ_LEN_MORE_CHUNKING` as the engine resolves it on a part with `dram` bytes."""
    with mock.patch.object(T, "_dram_total_bytes", lambda _d, b=dram: b), \
         mock.patch.object(T.ttnn, "get_max_worker_l1_unreserved_size",
                           lambda: T._WH_FULL_L1_PER_CORE), \
         mock.patch.dict("os.environ", {}, clear=False) as env:
        env.pop("TT_BIO_SEQ_LEN_MORE_CHUNKING", None)
        T._apply_grid_thresholds(grid)
        return T.SEQ_LEN_MORE_CHUNKING


def _pair_axis(residues):
    return bucketed_width(TOKENS[residues], TOKEN_BUCKET)


def test_the_gate_resolves_to_1088_tokens_on_a_12_gib_wormhole(restore_globals):
    assert _gate(WH_DRAM) == 1088
    assert _gate(BH_DRAM) == 1536      # and to the Blackhole baseline on a 32 GiB part


def test_the_published_ceiling_and_first_failure_straddle_the_gate(restore_globals):
    """The finding. `pass_at` is the last rung on the whole-tensor path, `fail_at` the first on
    the chunked one, for BOTH checkpoints and on two independently measured ladders."""
    gate = _gate(WH_DRAM)
    for model in ("opendde", "opendde-abag"):
        c = size_limits.CEILINGS[model]["wormhole_b0"]
        assert (c.pass_at, c.fail_at) == (544, 576)
        assert _pair_axis(c.pass_at) == gate            # 1088, exactly at it
        assert _pair_axis(c.fail_at) == 1120 > gate     # the first rung over


def test_no_other_rung_boundary_lands_on_the_gate(restore_globals):
    """The negative control: if several rung gaps contained the gate the coincidence would be
    worth nothing. Exactly one does, and it is the measured one."""
    gate = _gate(WH_DRAM)
    rungs = sorted(TOKENS)
    crossings = [(a, b) for a, b in zip(rungs, rungs[1:])
                 if _pair_axis(a) <= gate < _pair_axis(b)]
    assert crossings == [(544, 576)]


def test_a_32_gib_part_puts_its_own_crossing_at_896(restore_globals):
    """Same gate, different part: the prediction a Blackhole ladder would test, and the reason
    this is not a Wormhole-only constant to be nudged."""
    gate = _gate(BH_DRAM)
    rungs = sorted(TOKENS)
    assert [(a, b) for a, b in zip(rungs, rungs[1:])
            if _pair_axis(a) <= gate < _pair_axis(b)] == [(768, 896)]
