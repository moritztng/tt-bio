"""The OPM depth join repeats b's depth rows over the I batch, so it only runs while that
[I, J, S*D] bf16 copy fits OPM_DEPTH_JOIN_MAX_BYTES; larger folds keep the per-row loop."""
import pytest

from tt_bio import tenstorrent as T


@pytest.mark.parametrize("n,s", [(224, 2), (256, 8), (512, 2), (195, 8)])
def test_af2ig_and_small_folds_join(n, s):
    assert T._opm_depth_join_fits(n, n, s, 32)


@pytest.mark.parametrize("n,s", [(1536, 8), (1536, 2), (2048, 2), (512, 8), (1024, 2)])
def test_large_folds_keep_the_row_loop(n, s):
    assert not T._opm_depth_join_fits(n, n, s, 32)


def test_cap_boundary():
    # 512 x 512 x 4 x 32 bf16 is exactly 64 MiB.
    assert T._opm_depth_join_fits(512, 512, 4, 32)
    assert not T._opm_depth_join_fits(513, 512, 4, 32)


def test_counter_separates_a_dark_join_from_a_declining_one():
    """Both arms of an A/B run the per-row loop when the join declines, so their coords come out
    bit-identical and the cell grades nothing. The counter is what tells that apart from a real pass."""
    T.OPM_DEPTH_JOIN_STATS[:] = [0, 0]
    assert T._opm_depth_join_takes(256, 256, 4, 32)          # fits, several rows: joined
    assert not T._opm_depth_join_takes(256, 256, 1, 32)      # one row has nothing to join
    assert not T._opm_depth_join_takes(1536, 1536, 8, 32)    # over the cap: per-row loop
    assert T.OPM_DEPTH_JOIN_STATS == [1, 2]


def test_counter_is_off_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(T, "_OPM_DEPTH_JOIN", False)
    T.OPM_DEPTH_JOIN_STATS[:] = [0, 0]
    assert not T._opm_depth_join_takes(256, 256, 4, 32)
    assert T.OPM_DEPTH_JOIN_STATS == [0, 1]
