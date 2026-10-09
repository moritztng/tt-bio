"""transition_shard: the row height fills the grid exactly, and shapes it was not measured on keep their path."""
import pytest

pytest.importorskip("ttnn", reason="tenstorrent.py imports ttnn at module scope")

from tt_bio import tenstorrent as tt  # noqa: E402


@pytest.fixture(autouse=True)
def wormhole_grid(monkeypatch):
    monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", (8, 9))


@pytest.mark.parametrize("W,rows,grid", [
    (736, 9, (8, 9)),    # c730, the measured shape: 23 row tiles a core
    (256, 18, (8, 9)),
    (512, 9, (8, 9)),
    (1024, 5, (8, 8)),   # 32 tiles a row: no multiple of 9 fits, 8 rows of the grid do
    (1536, 3, (8, 9)),
])
def test_pair_rows_fill_the_grid(W, rows, grid):
    assert tt._transition_shard_rows(W, 256, 1024) == rows
    mt = rows * (-(-W // 32))
    assert tt._transition_shard_grid(mt, 32, 8) == grid
    assert mt // grid[1] <= tt._TRANSITION_SHARD_PM


def test_msa_transition_keeps_its_path():
    # 128 channels out are 4 tiles: no 8-column split, and 4x8 measured slower than interleaved.
    assert tt._transition_shard_rows(736, 128, 512) == 0
    assert tt._transition_shard_grid(368, 16, 4) is None


def test_a_block_that_does_not_fill_half_the_grid_declines():
    assert tt._transition_shard_grid(23 * 4, 32, 8) is None   # a 4-row remainder at W=736
    assert tt._transition_shard_grid(23 * 7, 32, 8) == (8, 7)


def test_blackhole_grid_takes_eight_columns(monkeypatch):
    monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", (11, 10))
    assert tt._transition_shard_rows(736, 256, 1024) == 10
    assert tt._transition_shard_grid(230, 32, 8) == (8, 10)


def test_lever_is_graded_into_both_modes():
    assert "transition_shard" in tt.LEVERS
    assert "transition_shard" not in tt.UNGRADED_LEVERS
    assert "transition_shard" in tt.NORMAL_LEVERS & tt.FAST_LEVERS
    assert "transition_shard" in tt.LATCH_STATS
