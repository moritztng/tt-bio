"""The host-concat byte budget is per part, and a 12 GiB part is byte-identical.

`CONCAT_HOST_BYTES_BASE` = 1.5 GiB was measured on a 12.0 GiB Wormhole Galaxy chip and shipped
unscaled to a 31.875 GiB Blackhole p150a, where it sent the OpenDDE refiner's pair channel join
to the host from 768 aa up: 6.55 s per call against a 9 ms device concat, 21.88x, ~15.6% of a
768 aa fold. Same defect class as tt-bio issue #11's L1 budgets -- a calibration point applied
outside its measured range -- so it gets the same table-driven per-part assertion, host-only, no
device, so it runs on every part forever.
"""
import pytest

import tt_bio.tenstorrent as T

GIB = 2 ** 30
WH_GALAXY = 12 * GIB                # measured: dram_peak's own printed total
P150A = 34_225_520_128              # measured: 8 banks x 4,278,190,016 B on qb1
BASE = 1536 * 2 ** 20               # 1.5 GiB


def test_base_is_the_measured_wormhole_figure():
    assert T.CONCAT_HOST_BYTES_BASE == BASE == 1_610_612_736


@pytest.mark.parametrize("dram,budget", [
    (WH_GALAXY, 1_610_612_736),     # 12.0 GiB // 8 IS the shipped constant: identical
    (P150A, 4_278_190_016),         # 31.875 GiB // 8 = 3.984 GiB
    (0, BASE),                      # no device / failed read: the base figure
])
def test_per_part_budget(dram, budget):
    assert T._concat_host_budget(dram) == budget


@pytest.mark.parametrize("dram", [0, 1, GIB, 8 * GIB, WH_GALAXY - 1, WH_GALAXY])
def test_never_tightens_below_the_measured_base(dram):
    """max() pins every part at or above today's figure, so no part can regress into
    MORE host concat than it does now."""
    assert T._concat_host_budget(dram) == BASE


def test_monotonic_in_dram():
    seq = [T._concat_host_budget(d) for d in range(0, 64 * GIB, GIB)]
    assert seq == sorted(seq)


@pytest.mark.parametrize("dram,host", [(WH_GALAXY, True), (P150A, False)])
def test_the_opendde_refiner_768aa_shape_splits_by_part(dram, host):
    """The behaviour, not just the number: the 768 aa refiner pair tensor (H=1494, c_z=384,
    bf16) is 1.5965 GiB, which is over a 12 GiB part's budget and under a p150a's."""
    pair_bytes = 1494 * 1494 * 384 * 2
    assert pair_bytes == 1_714_203_648
    assert (pair_bytes > T._concat_host_budget(dram)) is host


def test_640aa_control_is_on_the_device_path_everywhere():
    """H=1243 = 1.1051 GiB is under both budgets, so the fix must be inert at 640 aa."""
    pair_bytes = 1243 * 1243 * 384 * 2
    for dram in (WH_GALAXY, P150A):
        assert pair_bytes < T._concat_host_budget(dram)


def test_env_override_wins_and_is_read_through_the_function(monkeypatch):
    monkeypatch.setattr(T, "_CONCAT_HOST_BYTES", None)
    monkeypatch.setenv("TT_BIO_CONCAT_HOST_BYTES", "12345")
    assert T.concat_host_bytes() == 12345


def test_budget_read_never_opens_a_device(monkeypatch):
    """_dram_total_bytes must not call get_device(): a byte test that brings a card up as a
    side effect would open a device during import-time model construction."""
    monkeypatch.setattr(T, "_device", None)
    monkeypatch.setattr(T, "get_device", lambda *a, **k: pytest.fail("opened a device"))
    assert T._dram_total_bytes() == 0
    monkeypatch.setattr(T, "_CONCAT_HOST_BYTES", None)
    monkeypatch.delenv("TT_BIO_CONCAT_HOST_BYTES", raising=False)
    assert T.concat_host_bytes() == BASE


def test_opendde_holds_no_frozen_copy_of_the_budget():
    """The regression guard no fold-level check can see: tt_bio.opendde used to
    `from .tenstorrent import CONCAT_HOST_BYTES`, an int bound at import time before any
    device is open. Resolving the budget at device-configure time would have left that site
    at 1.5 GiB forever, silently, with half the fix dead."""
    import tt_bio.opendde as O
    assert not any(isinstance(getattr(O, n, None), int) and n.isupper() and "CONCAT" in n
                   for n in dir(O)), "opendde must reach the budget through the function"
    assert O.concat_host_bytes is T.concat_host_bytes


# The Wormhole total is a printed figure, not a byte count: dram_peak formats it as "%.1f GiB"
# (perf/whb2/out/cap_wh/cap_unchunked_1024.dram, a WH Galaxy artifact, reads "of 12.0 GiB"), so
# the true total is anywhere in [11.95, 12.05) GiB. That window is what stands between the
# arithmetic identity above and a claim about real Wormhole behaviour.
WH_PRINT_WINDOW = (int(11.95 * GIB), int(12.05 * GIB) - 1)


@pytest.mark.parametrize("dram", [WH_PRINT_WINDOW[0], WH_GALAXY, WH_PRINT_WINDOW[1]])
def test_wormhole_verdicts_hold_across_the_printed_precision(dram):
    """Wormhole behaviour is unchanged for any total its own printed figure admits.

    Below 12.0 GiB the budget clamps to BASE and nothing moves. Above it the budget can widen,
    but only by the slack in the last printed digit: at most 0.42%, which shifts the host/device
    crossing by under two units of H (~1 residue at H = 1.945 x aa) and only ever toward MORE
    device concat. Both shapes the release gate pins keep their Wormhole verdict across the
    whole window, so no Wormhole fold changes path -- which is the claim two production folds on
    the shared Galaxy would have been buying."""
    budget = T._concat_host_budget(dram)
    assert BASE <= budget <= BASE * 1.005
    assert 1494 * 1494 * 384 * 2 > budget      # 768 aa refiner: host on Wormhole, as today
    assert 1243 * 1243 * 384 * 2 < budget      # 640 aa control: device on Wormhole, as today
