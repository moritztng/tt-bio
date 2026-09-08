"""The ending transpose's L1 gate has to leave its layer_norm room, at every shape.

The bug this pins is not a capacity wall. `_TRANSPOSE_L1_HEADROOM = 1.25` prices the consumer's
static circular buffers as a fraction of the block, so its own best case leaves 293 216 B per core
free while the consumer needs 349 184 B (measured; tests/test_l1_clash_census.py). The top of the
gate's admitted range is therefore broken at every shape and nothing below it is, which is why
576 aa throws and 608 aa folds. Numbers here are the Galaxy Wormhole 8x9 grid, card 2, 2026-09-08.

Every `width` below is a PAIR-TOKEN count, not a residue count, and the two are not close.
`build_structural_token_features` emits a backbone and a sidechain structural token per non-GLY
residue, so OpenDDE's pair axis runs at 1.945x its residue count -- measured on the rung
sequences: 128 aa -> 249 tokens, 512 -> 995, 544 -> 1057, 576 -> 1120, 640 -> 1243, 1024 -> 1993,
each then bucketed up to a multiple of 32. `test_the_pair_axis_is_not_the_residue_axis` pins it.
The first reading of this bug used the residue count and so named a [480, 672, 128] block for a
throw whose 82 575 360 B is 288 x 1120 x 128 bf16.
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


def test_the_broken_band_is_one_bucket_wide_and_no_rung_is_in_it(wh):
    """The band is real and narrow, and at chunk 480 no rung's pair width reaches it.

    Enumerated over the whole pair axis, to 2048: `1024 aa` is 2016 pair tokens, so a range that
    stops at 1056 stops halfway. Exactly one multiple of 32 is admitted by the 1.25 gate and too
    big to leave the consumer its buffers -- 672 tokens -- and that is 345 residues, which is not
    a rung and not the 576 aa throw. So the throw is NOT a block at chunk 480: 576 aa is 1120
    tokens and 480 x 1120 x 128 bf16 is 137 625 600 B, which misses L1 by 63 %.
    """
    old = int(WH_L1_PER_CORE * WH_CORES / tt.TRANSPOSE_L1_HEADROOM)
    safe = (WH_L1_PER_CORE - LN_CB_NEED) * WH_CORES  # volume that still leaves the consumer room
    band = [n for n in range(32, 2049, 32) if safe < 122880 * n <= old]   # 122880 = 480 x 128 x 2
    assert band == [672]
    assert 122880 * 640 <= safe and 122880 * 704 > old
    assert 480 * 1120 * 128 * 2 > old                 # 576 aa at chunk 480 never reaches L1


def test_the_new_gate_leaves_the_consumer_room_at_every_admitted_shape(wh):
    reserve = tt._pair_bias_ln_reserve()
    assert reserve == 384 * 1024
    assert reserve > LN_CB_NEED                     # 1.13x of the measurement
    free = WH_L1_PER_CORE - admitted(WH_L1_PER_CORE, WH_CORES, reserve) // WH_CORES
    assert free == reserve                          # headroom 1.0: the reserve IS the guarantee
    assert free >= LN_CB_NEED


def test_the_gate_and_the_cap_agree_on_what_reaches_l1(wh):
    """The reserve refuses; the cap then names a height that fits. Both, or the route is lost.

    Stated over the rungs' real pair widths. Every one of them is chunked (`SEQ_LEN_MORE_CHUNKING`
    is 608 on this part and the smallest rung here is 1024 tokens), and at chunk 480 every one of
    them misses L1 under either gate -- which is why the cap, not the reserve, is what recovers
    the route.
    """
    reserve = tt._pair_bias_ln_reserve()
    cap = admitted(WH_L1_PER_CORE, WH_CORES, reserve)
    for width in (1024, 1088, 1120, 1248, 1504, 1760, 2016):     # 512..1024 aa
        assert 480 * width * 128 * 2 > cap                        # chunk 480 never fits
        rows = tt._pair_bias_ln_row_cap(width, 128)
        assert 0 < rows < 480                                     # so the cap always binds
        blk = width * rows * 128 * 2
        assert blk <= cap                                         # and what it names does fit
        assert WH_L1_PER_CORE - blk // WH_CORES >= LN_CB_NEED


@pytest.mark.parametrize("width,expect", [(672, 448), (1120, 256), (2016, 128), (512, 576)])
def test_row_cap_keeps_the_l1_route_instead_of_abandoning_it(wh, width, expect):
    cap = tt._pair_bias_ln_row_cap(width, 128)
    assert cap == expect
    if cap < 480:                                   # only then does it change a chunk boundary
        blk = width * cap * 128 * 2
        assert blk <= admitted(WH_L1_PER_CORE, WH_CORES, tt._pair_bias_ln_reserve())
        assert WH_L1_PER_CORE - blk // WH_CORES >= LN_CB_NEED
    else:
        assert cap > tt.TRIANGLE_ATT_CHUNK_SIZE_FAST or cap >= 480   # does not bind below 640


def test_1024_aa_keeps_the_l1_route(wh):
    """The ceiling this exists for. 1024 residues are 2016 pair tokens, and 128 rows of them fit."""
    cap = tt._pair_bias_ln_row_cap(2016, 128)
    assert cap == 128
    blk = 2016 * cap * 128 * 2
    assert blk <= admitted(WH_L1_PER_CORE, WH_CORES, tt._pair_bias_ln_reserve())
    assert WH_L1_PER_CORE - blk // WH_CORES >= LN_CB_NEED
    # Uncapped, the 2016-token block at chunk 480 is 247 726 080 B and misses L1 by 193 %, so the
    # route is something the cap RECOVERS, not something it costs.
    assert 2016 * 480 * 128 * 2 > WH_L1_PER_CORE * WH_CORES / tt.TRANSPOSE_L1_HEADROOM


def test_blackhole_does_not_move(monkeypatch):
    """Negative control: on a >= 110-core grid the reserve is 0 and the gate is the old one."""
    monkeypatch.setattr(ttnn, "get_max_worker_l1_unreserved_size", lambda: BH_L1_PER_CORE)
    monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", (11, 10))
    monkeypatch.setattr(tt, "_IS_SMALL_GRID", False)
    assert tt._pair_bias_ln_reserve() == 0
    assert tt._pair_bias_ln_row_cap(672, 128) == 0          # 0 means "no cap", not "cap of 0"
    assert tt._pair_bias_ln_row_cap(2016, 128) == 0


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
    that band and it opens at 561 pair tokens and shuts at 574: no multiple of 32 is inside, so the
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


def test_the_pair_axis_is_not_the_residue_axis():
    """OpenDDE's pair track runs at 1.945x its residue count, and every byte budget here is on it.

    MEASURED on the rung sequences with `build_structural_token_features`, host-only, 2026-09-08:
    a backbone token plus a sidechain token per non-GLY residue. Pinned because reading a residue
    count as a pair width is what named the wrong block for the 576 aa throw, and nothing in the
    log distinguishes the two -- both factor 82 575 360 B exactly.
    """
    from tt_bio.token_axis import bucketed_width
    measured = {128: 249, 256: 497, 512: 995, 544: 1057, 576: 1120,
                608: 1183, 640: 1243, 768: 1494, 896: 1744, 1024: 1993}
    for res, ns in measured.items():
        assert 1.93 < ns / res < 1.96, (res, ns)
    assert bucketed_width(measured[576], 32) == 1120
    assert bucketed_width(measured[1024], 32) == 2016
    # The two readings of the throw's 82 575 360 B, and why the axis has to be measured:
    assert 480 * 672 * 128 * 2 == 82575360          # the residue-count reading, wrong
    assert 288 * 1120 * 128 * 2 == 82575360         # the pair-token reading, 576 aa
