"""k1_linear's Wormhole plan never asks for more circular buffer than a core has.

8pe3-assembly1 (Mol-1, 792 tokens) threw at its first diffusion step on a Wormhole Galaxy chip:
"circular buffers ... grow to 1500448 B which is beyond max L1 size of 1499136 B". The plan was
priced against a constant 1.4 MB, but the CB region starts at 103712 B (the firmware map plus the
69 KiB kernel-config ring), so only 1395424 B are free for CBs, the same number the L1 allocator
reports per bank. These tests pin the price, the budget and the planner over every shape the
Protenix family reaches on Wormhole. No device: `k1_plan` is pure arithmetic.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ttnn = pytest.importorskip("ttnn")

import tt_bio.tenstorrent as T  # noqa: E402

WH_L1_TOP = 1499136     # "max L1 size" in the throw
WH_CB_BASE = 103712     # MEM_MAP_END 33056 + 69 KiB kernel-config ring (ttnn 0.68 wh_hal_tensix.cpp)
WH_BANK = 1395424       # per-bank L1 the allocator reports on a 12 GiB Wormhole Galaxy chip
F32, BF16, B8 = 4096, 2048, 1088
ACC = 4096              # a non-fp32 output gets an fp32 interm CB beside it

# Every operand/output/bias format combination k1_linear can be handed.
TILES = [(a, b, o, bias) for a in (F32, BF16, B8) for b in (F32, BF16, B8)
         for o in (F32, BF16 + ACC, B8 + ACC) for bias in (0, F32, BF16, B8)]
GRIDS = [(8, 9), (8, 8)]   # Wormhole Galaxy, n150 / n300
# Rows a Wormhole fold reaches: 5 diffusion samples x <= 24 heavy atoms a token (an RNA G is 23) x
# 1536 tokens, past every Wormhole cap (Mol-1 896, OpenDDE 1088). Widths: c_atom 128 and its
# 2x / 4x transitions, c_token 768 and its 2x / 4x, OpenFold3's 64-bin distogram, and every width
# to 128 tiles for the row counts swept coarsely.
MT_MAX = 5 * 24 * 1536 // 32
NT_REACHED = (2, 4, 8, 16, 24, 48, 96)


def test_the_price_reproduces_the_8pe3_throw():
    """fp32 128 -> 128 atom linear, no bias, 113 rows a core: ttnn's own number, to the byte."""
    assert WH_CB_BASE + T.k1_cb_bytes(113, 1, (F32, F32, F32, 0)) == 1500448
    assert WH_L1_TOP - WH_CB_BASE == WH_BANK


def test_8pe3_plan_now_fits_in_two_blocks():
    mt = 9 * 113                                   # per_core_M 113 on an 8x9 grid
    tiles = (F32, F32, F32, 0)
    h, w, oh, pm, pn = T.k1_plan(mt, 4, 8, 9, tiles, WH_BANK)
    assert (oh, pm, pn) == (57, 114, 1)            # not 113 one-row blocks: 113 is prime
    assert T.k1_cb_bytes(oh, pn, tiles) <= WH_BANK


def _old_plan(mt, nt, gx, gy, tiles):
    """The rule before the fix, budget aside: the tallest block dividing per_core_M."""
    pm, pn = -(-mt // gy), -(-nt // gx)
    h, w = max(((h, w) for h in range(1, min(pm, 4) + 1) for w in range(1, min(pn, 4) + 1)
                if pm % h == 0 and pn % w == 0 and h * w <= 4), key=lambda s: (s[0] * s[1], s[1]))
    oh = max((d for d in range(h, pm + 1, h) if pm % d == 0 and T.k1_cb_bytes(d, pn, tiles) <= WH_BANK),
             default=None)
    return h, w, oh, pm, pn


def _check(mt, nt, gx, gy, tiles):
    plan = T.k1_plan(mt, nt, gx, gy, tiles, WH_BANK)
    assert plan is not None, (mt, nt, gx, gy, tiles)
    h, w, oh, pm, pn = plan
    assert T.k1_cb_bytes(oh, pn, tiles) <= WH_BANK, (mt, nt, tiles, plan)
    assert oh % h == 0 and pm % oh == 0 and pn % w == 0 and h * w <= 4
    assert pm * gy >= mt and -(-mt // pm) <= gy            # every row has a core
    oh0, pm0 = _old_plan(mt, nt, gx, gy, tiles)[2:4]
    if oh0 == pm0:                                           # one block fitted: same program as before
        assert (oh, pm) == (pm0, pm0)
    else:                                                    # never more blocks than the divisor rule
        assert pm // oh <= pm0 // oh0


@pytest.mark.parametrize("gx,gy", GRIDS)
def test_every_reached_shape_has_a_plan_that_fits(gx, gy):
    for tiles in TILES:
        for nt in NT_REACHED:
            for pm in range(1, -(-MT_MAX // gy) + 1):
                _check(pm * gy, nt, gx, gy, tiles)           # a full last core
                _check(pm * gy - gy + 1, nt, gx, gy, tiles)  # one row on the last core


@pytest.mark.parametrize("gx,gy", GRIDS)
def test_every_width_to_128_tiles_has_a_plan_that_fits(gx, gy):
    for tiles in TILES:
        for nt in range(1, 129):
            for mt in range(1, MT_MAX + 1, 97):
                _check(mt, nt, gx, gy, tiles)
