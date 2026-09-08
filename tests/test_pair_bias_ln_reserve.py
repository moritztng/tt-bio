"""The ending transpose's L1 gate has to leave its layer_norm room, at every shape.

The bug this pins is not a capacity wall. `_TRANSPOSE_L1_HEADROOM = 1.25` prices the consumer's
static circular buffers as a fraction of the block, so its own best case leaves 293 216 B per core
free while the consumer needs 349 184 B (measured; tests/test_l1_clash_census.py). The top of the
gate's admitted range is therefore broken at every shape and nothing below it is, which is why
576 aa throws and 608 aa folds. Numbers here are the Galaxy Wormhole 8x9 grid, card 2, 2026-09-08.
"""
import pytest

pytest.importorskip("ttnn", reason="tenstorrent.py imports ttnn at module scope")

import ttnn  # noqa: E402

from tt_bio import tenstorrent as tt  # noqa: E402

WH_L1_PER_CORE = 1466080
WH_CORES = 72                    # 8x9, read off the throw's own core range
BH_L1_PER_CORE = 1532448
BH_CORES = 110                   # 11x10
LN_CB_NEED = 349184              # measured at [480, 672, 128] bf16 on Wormhole


@pytest.fixture
def wh(monkeypatch):
    """A Wormhole small grid, without a device. Restored by monkeypatch."""
    monkeypatch.setattr(ttnn, "get_max_worker_l1_unreserved_size", lambda: WH_L1_PER_CORE)
    monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", (8, 9))
    monkeypatch.setattr(tt, "_IS_SMALL_GRID", True)
    monkeypatch.setattr(tt, "_PAIR_BIAS_LN_CONSUMER_RESERVE", tt.PAIR_BIAS_LN_CONSUMER_RESERVE)
    monkeypatch.setattr(tt, "_PAIR_BIAS_LN_CAP", True)


def admitted(per_core, cores, reserve):
    """The largest block volume in bytes the byte-budget gate lets into L1."""
    return (per_core - reserve) * cores


def test_the_old_multiplicative_gate_admits_what_cannot_fit(wh):
    """The premise. Without this the fix has nothing to fix."""
    biggest = int(WH_L1_PER_CORE * WH_CORES / tt.TRANSPOSE_L1_HEADROOM)
    assert biggest // WH_CORES == 1172864
    assert WH_L1_PER_CORE - biggest // WH_CORES == 293216
    assert 293216 < LN_CB_NEED                      # short by 55968 B at EVERY shape
    # And the 576 aa block is inside that range: admitted, then throws at the consumer.
    assert 480 * 672 * 128 * 2 <= biggest


def test_the_broken_band_holds_exactly_one_bucket(wh):
    """Non-monotonicity, derived: 672 structural tokens land in it and no other multiple of 32."""
    old = int(WH_L1_PER_CORE * WH_CORES / tt.TRANSPOSE_L1_HEADROOM)
    safe = (WH_L1_PER_CORE - LN_CB_NEED) * WH_CORES  # volume that still leaves the consumer room
    band = [n for n in range(32, 1057, 32) if safe < 122880 * n <= old]   # 122880 = 480 x 128 x 2
    assert band == [672]
    # 640 fits under the old gate on 24 640 B/core of slack; 704 misses L1 entirely and so runs
    # DRAM, where there is no clash. A bigger input succeeding is the gate's boundary, not a
    # paradox.
    assert 122880 * 640 <= safe and 122880 * 704 > old


def test_the_new_gate_leaves_the_consumer_room_at_every_admitted_shape(wh):
    reserve = tt._pair_bias_ln_reserve()
    assert reserve == 384 * 1024
    assert reserve > LN_CB_NEED                     # 1.13x of the measurement
    free = WH_L1_PER_CORE - admitted(WH_L1_PER_CORE, WH_CORES, reserve) // WH_CORES
    assert free == reserve                          # headroom 1.0: the reserve IS the guarantee
    assert free >= LN_CB_NEED


def test_the_new_gate_refuses_the_576_block_and_keeps_the_544_one(wh):
    reserve = tt._pair_bias_ln_reserve()
    cap = admitted(WH_L1_PER_CORE, WH_CORES, reserve)
    assert 480 * 672 * 128 * 2 > cap                # the throw: L1 refused, so no clash
    # 544 aa takes the non-chunked branch (S <= SEQ_LEN_MORE_CHUNKING) with the whole 544-token
    # pair tensor, which fits today and must still fit. This is what rules out pricing the
    # reserve on top of 1.25 as well.
    assert 544 * 544 * 128 * 2 <= cap
    assert 544 * 544 * 128 * 2 * tt.TRANSPOSE_L1_HEADROOM > cap


@pytest.mark.parametrize("width,expect", [(672, 448), (640, 448), (1024, 288), (512, 576)])
def test_row_cap_keeps_the_l1_route_instead_of_abandoning_it(wh, width, expect):
    cap = tt._pair_bias_ln_row_cap(width, 128)
    assert cap == expect
    if cap < 480:                                   # only then does it change a chunk boundary
        blk = width * cap * 128 * 2
        assert blk <= admitted(WH_L1_PER_CORE, WH_CORES, tt._pair_bias_ln_reserve())
        assert WH_L1_PER_CORE - blk // WH_CORES >= LN_CB_NEED
    else:
        assert cap > tt.TRIANGLE_ATT_CHUNK_SIZE_FAST or cap >= 480   # does not bind below 640


def test_1024_tokens_keep_the_l1_route(wh):
    """The ceiling this exists for. 288 rows of a 1024-token block, still in L1."""
    cap = tt._pair_bias_ln_row_cap(1024, 128)
    blk = 1024 * cap * 128 * 2
    assert blk <= admitted(WH_L1_PER_CORE, WH_CORES, tt._pair_bias_ln_reserve())
    assert WH_L1_PER_CORE - blk // WH_CORES >= LN_CB_NEED
    # Without the cap the 1024 block is 125 829 120 B and misses L1 by 19 %, so the route is
    # something the cap RECOVERS, not something it costs.
    assert 1024 * 480 * 128 * 2 > WH_L1_PER_CORE * WH_CORES / tt.TRANSPOSE_L1_HEADROOM


def test_blackhole_does_not_move(monkeypatch):
    """Negative control: on a >= 110-core grid the reserve is 0 and the gate is the old one."""
    monkeypatch.setattr(ttnn, "get_max_worker_l1_unreserved_size", lambda: BH_L1_PER_CORE)
    monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", (11, 10))
    monkeypatch.setattr(tt, "_IS_SMALL_GRID", False)
    assert tt._pair_bias_ln_reserve() == 0
    assert tt._pair_bias_ln_row_cap(672, 128) == 0          # 0 means "no cap", not "cap of 0"
    assert tt._pair_bias_ln_row_cap(1024, 128) == 0


def test_cap_off_switch_leaves_the_gate_on(wh, monkeypatch):
    """The parity A/B arm: the cap off, the reserve still in. Anything else is two changes."""
    monkeypatch.setattr(tt, "_PAIR_BIAS_LN_CAP", False)
    assert tt._pair_bias_ln_row_cap(672, 128) == 0
    assert tt._pair_bias_ln_reserve() == 384 * 1024


def test_reserve_off_switch_restores_the_old_gate(wh, monkeypatch):
    """The base arm for the A/B: reserve 0 turns off both halves at once."""
    monkeypatch.setattr(tt, "_PAIR_BIAS_LN_CONSUMER_RESERVE", 0)
    assert tt._pair_bias_ln_reserve() == 0
    assert tt._pair_bias_ln_row_cap(672, 128) == 0


def test_row_cap_is_a_multiple_of_the_tile_and_of_the_token_bucket(wh):
    for w in range(32, 1057, 32):
        cap = tt._pair_bias_ln_row_cap(w, 128)
        assert cap % 32 == 0, w
        assert cap > 0, w


def test_the_non_chunked_geometry_never_lands_in_the_broken_band(wh):
    """The one path the fix does not touch, and why it does not have to.

    `TriangleAttention.__call__`'s non-chunked ending branch still asks for the whole pair tensor
    through `_transpose_memory_config`, with no reserve and no retry. It runs when the pair width
    is at or below `SEQ_LEN_MORE_CHUNKING`, which `_apply_grid_thresholds` snaps to 608 on this
    part, and its volume is 256 x Nsw^2 rather than the chunked path's 122 880 x Nsw. Solve for
    that band and it opens at Nsw 561 and shuts at 574: no multiple of 32 is inside, so the
    branch is safe by arithmetic. Pinned because "safe by arithmetic" stops being true the moment
    the bucket, the channel count or the threshold moves, and then it should fail here rather
    than in a fold.
    """
    old = int(WH_L1_PER_CORE * WH_CORES / tt.TRANSPOSE_L1_HEADROOM)
    safe = (WH_L1_PER_CORE - LN_CB_NEED) * WH_CORES
    band = [n for n in range(32, 641, 32) if safe < 256 * n * n <= old]
    assert band == []
    # The two neighbours that bracket it, so the test is not vacuous on a shifted band.
    assert 256 * 544 * 544 <= safe                  # admitted to L1 and the consumer fits
    assert 256 * 576 * 576 > old                    # refused L1, runs DRAM, no clash
