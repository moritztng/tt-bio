"""The protenix template embedder's L1-resident pair tensor, and why it does not starve.

`Protenix._ln(..., l1=True)` (tt_bio/protenix.py) is the third call site of the pattern that
crashed Boltz-2 at 704 padded tokens: `_l1_layer_norm(x, 1.5)`, a headroom multiple priced
against the whole grid's aggregate L1, guarding a pair tensor that a narrow projection then
reads in place. `test_pair_l1_consumer_reserve.py` covers the two sites that had to be fixed.
This one covers the site that measured out safe, so that "safe" is a pinned fact and not a
memory: it holds while the numbers below hold, and the tests fail the moment one moves.

Measured on a p150a, 13x10, `perf/protenix_tpl_l1/` (probe.py and fold_l1_trace.py):

  * the window opens with L1 EMPTY -- 1461760 B/bank free at every one of the 10 recycling
    cycles of a real 496-token fold. Boltz-2's crash needed 247 KB/core of other live buffers
    on top of the pair tensor; here there are none.
  * at 506 tokens, the largest shape the gate admits, the normed tensor takes 1021952 B/bank
    and the projection's L1 output another 256000, and program creation still leaves 183808.
  * the whole window degrades instead of throwing. Starved with synthetic ballast the order is
    L1 output -> DRAM output (caught in `_narrow_proj_linear`) -> DRAM norm (caught in
    `_l1_layer_norm`), and the norm stops taking L1 while 194048 B/bank are still free, which
    is 3.7x what the DRAM-output arm's circular buffers need.

Host-only: allocator arithmetic against pinned part figures, no device and no fold.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tt_bio import protenix as P
from tt_bio import tenstorrent as T

# p150a, measured: what the gate reads, and what the allocator actually has per bank.
GATE_PER_CORE = 1532416
BANK_BYTES = 1461760
# The headroom `Protenix._ln` passes, and the pair widths that reach it: Protenix-v2's c_z=256
# and OpenDDE's 384 (tt_bio/opendde.py builds the same Trunk at c_z=384).
HEADROOM = 1.5
C_Z = (256, 384)
# Every grid tt-bio ships on. On a real part the L1 banks ARE the worker cores, which is what
# makes the margin below grid-independent.
GRIDS = ((13, 10), (11, 10), (8, 9))
# Measured B/bank the consumers need at the worst admitted shape: 256000 for the projection's
# own L1 output plus its static circular buffers, which fit in the 52736 that were left. The
# L1-output arm was refused at 275968 free and taken at 308736, so the need is in that interval
# and this is its upper end.
MEASURED_CONSUMER_NEED = 308736


@pytest.fixture
def part(monkeypatch):
    """Pin the p150a figures so the arithmetic is the test, not the host."""
    monkeypatch.setattr(T.ttnn, "get_max_worker_l1_unreserved_size", lambda: GATE_PER_CORE)


def pair(tokens: int, c_z: int):
    """The pair tensor as `Trunk._template` hands it to the gate: (1, N, N, c_z), bf16."""
    return SimpleNamespace(shape=(1, tokens, tokens, c_z), dtype=T.ttnn.bfloat16)


def admits(tokens: int, c_z: int, headroom: float = HEADROOM, reserve: int = 0) -> bool:
    mc = T._l1_memory_config_if_it_fits(pair(tokens, c_z), headroom, reserve)
    return mc is T.ttnn.L1_MEMORY_CONFIG


def worst_admitted(c_z: int, **kw) -> int:
    """The largest token count the gate lets into L1 -- where the consumer has least room."""
    return max(n for n in range(32, 2048) if admits(n, c_z, **kw))


def bank_bytes(tokens: int, c_z: int, banks: int) -> int:
    """The normed tensor's share of one L1 bank, as the allocator interleaves it."""
    return tokens * (((tokens + 31) // 32) * 32) * c_z * 2 // banks


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("c_z", C_Z)
def test_the_consumers_have_more_room_than_they_need(part, monkeypatch, grid, c_z):
    """The whole point: at the worst shape the gate admits, the consumers still fit.

    Measured, not argued -- 506 tokens on a 13x10 leaves 439808 B/bank against a 308736 need,
    and the 11x10 and 8x9 config paths were run at their own worst shapes with the free L1
    matched to a real part of that size (perf/protenix_tpl_l1/results.md).
    """
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", grid)
    n = worst_admitted(c_z)
    free = BANK_BYTES - bank_bytes(n, c_z, grid[0] * grid[1])
    assert free >= MEASURED_CONSUMER_NEED, (grid, c_z, n, free)


def test_the_headroom_is_what_leaves_that_room(part, monkeypatch):
    """Why the margin holds on a part nobody has measured yet.

    A headroom of h admits at most per_core/h bytes per bank, so it always leaves
    per_core * (1 - 1/h) free -- 510805 B here, independent of grid, token count and pair
    width. That is the number the consumers have to fit inside, and they need 308736, so the
    margin is 1.65x. The danger is LOWERING it: below 1.2524 the guaranteed free drops under the
    measured need and this site becomes the Boltz-2 crash. `TRANSPOSE_L1_HEADROOM` was cut
    2.5 -> 1.25 for perf on the same helper, which is the pressure that would do it.
    """
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", (13, 10))
    guaranteed = int(GATE_PER_CORE * (1 - 1 / HEADROOM))
    assert guaranteed >= MEASURED_CONSUMER_NEED
    assert admits(worst_admitted(256), 256)


@pytest.mark.parametrize("c_z,boundary", [(256, 506), (384, 415)])
def test_the_admission_boundary_is_where_it_was_measured(part, monkeypatch, c_z, boundary):
    """Pins the boundary the device measurements were taken at: 506 tokens at Protenix-v2's
    c_z=256, 415 at OpenDDE's 384 (probe.py confirmed L1 at 415 and DRAM at 416)."""
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", (13, 10))
    assert worst_admitted(c_z) == boundary


def test_the_decision_is_monotone_in_token_count(part, monkeypatch):
    """No second window can open at a larger shape, so one measurement at the boundary covers
    every shape below it. Same argument as the Boltz-2 sites: area against a fixed budget."""
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", (13, 10))
    seen_dram = False
    for n in range(64, 1200, 8):
        if not admits(n, 256):
            seen_dram = True
        else:
            assert not seen_dram, f"L1 came back at {n} after a refusal"


def test_copying_the_boltz2_reserve_here_would_cost_shapes_that_fold_today(part, monkeypatch):
    """The trap this file exists to stop.

    `_PAIR_L1_CONSUMER_RESERVE` is 640 KiB, measured for a softmax that needs 563658 B/core.
    Passing it here would move the boundary from 506 tokens to 463 and take the L1 lever off
    every fold in between -- a real perf loss to fix a crash this site does not have. Only a
    reserve in a 2016-byte window reproduces today's decision exactly, which is another way of
    saying the 1.5 multiplier already IS the reserve at this shape.
    """
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", (13, 10))
    assert worst_admitted(256) == 506
    assert worst_admitted(256, headroom=1.0, reserve=T._PAIR_L1_CONSUMER_RESERVE) == 463


def test_the_residency_window_is_still_the_projections_only(part):
    """The template embedder ALREADY had a clash of this class (program 173), and it was fixed
    by freeing the tensor before the block loop rather than by the budget. If the deallocate
    moves back below the loop, the two PairformerLayers per template are inside the window
    again and their trimul is a consumer nobody measured.
    """
    src = Path(P.__file__).read_text()
    tpl = src[src.index("def _template("):]
    tpl = tpl[:tpl.index("\n    def ")]
    assert tpl.index("ttnn.deallocate(zn)") < tpl.index("for t in range(nt):")
    assert "_l1_layer_norm(x, 1.5" in src   # the call these numbers were measured at
