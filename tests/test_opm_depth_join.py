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
