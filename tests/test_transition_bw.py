"""transition_bw: one K block per (op, K), whatever the grid, and a config on every grid the fold runs on."""
import pytest

pytest.importorskip("ttnn", reason="tenstorrent.py imports ttnn at module scope")

from tt_bio import tenstorrent as tt  # noqa: E402


@pytest.fixture(autouse=True)
def budget(monkeypatch):
    monkeypatch.setattr(tt, "_matmul_cb_budget", lambda: 1_400_000)
    tt._transition_bw_config.cache_clear()
    yield
    tt._transition_bw_config.cache_clear()


GRIDS = [(8, 9), (8, 8), (7, 9), (11, 10), (13, 10)]
# (op, row tiles, K tiles, N tiles, bfp8, silu): Protenix pair / MSA c730 blocks, OpenDDE c_z 384, Boltz-2 MSA c 64
SHAPES = [("fc1", 115, 8, 32, False, True), ("fc2", 115, 8, 32, False, False), ("fc3", 115, 32, 8, False, False),
          ("fc1", 368, 4, 16, False, True), ("fc3", 368, 16, 4, False, False), ("fc3", 207, 32, 8, True, False),
          ("fc1", 46, 12, 48, False, True), ("fc3", 46, 48, 12, False, False), ("fc3", 368, 8, 2, False, False)]


@pytest.mark.parametrize("key", SHAPES)
def test_k_block_is_the_shapes_on_every_grid(key, monkeypatch):
    blocks = set()
    for g in GRIDS:
        monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", g)
        tt._transition_bw_config.cache_clear()
        cfg = tt._transition_bw_config(*key)
        assert cfg is not None, g
        assert cfg.out_subblock_h * cfg.out_subblock_w <= 4
        assert cfg.per_core_M % cfg.out_block_h == 0
        blocks.add(cfg.in0_block_w)
    assert blocks == {tt._transition_kb(key[0], key[2])}


def test_k_blocks():
    assert [tt._transition_kb("fc1", k) for k in (4, 8, 12)] == [4, 8, 4]
    assert [tt._transition_kb("fc3", k) for k in (8, 16, 32, 48)] == [2, 2, 2, 2]
    assert tt._transition_kb("fc3", 3) == 1


def test_narrow_output_splits_rows_wide_output_splits_both(monkeypatch):
    monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", (8, 9))
    assert isinstance(tt._transition_bw_config("fc3", 115, 32, 8, False, False),
                      tt.ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig)
    cfg = tt._transition_bw_config("fc1", 115, 8, 32, False, True)
    assert isinstance(cfg, tt.ttnn.MatmulMultiCoreReuseMultiCastProgramConfig)
    assert cfg.fused_activation is not None
    assert tt._transition_bw_config("fc2", 115, 8, 32, False, False).fused_activation is None


def test_lever_is_fast_only():
    assert "transition_bw" in tt.FAST_LEVERS and "transition_bw" not in tt.NORMAL_LEVERS
