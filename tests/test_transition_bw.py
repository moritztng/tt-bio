"""transition_bw: the measured K block lands on the c730 row blocks and nowhere it was not measured."""
import pytest

pytest.importorskip("ttnn", reason="tenstorrent.py imports ttnn at module scope")

from tt_bio import tenstorrent as tt  # noqa: E402


@pytest.fixture(autouse=True)
def wormhole_grid(monkeypatch):
    monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", (8, 9))
    monkeypatch.setattr(tt, "_matmul_cb_budget", lambda: 1_400_000)
    tt._transition_bw_config.cache_clear()
    yield
    tt._transition_bw_config.cache_clear()


@pytest.mark.parametrize("key,bw,one_d", [
    (("fc1", 115, 8, 32, False, True), 8, False),
    (("fc2", 115, 8, 32, False, False), 8, False),
    (("fc2", 368, 4, 16, False, False), 2, False),
    (("fc3", 368, 16, 4, False, False), 2, True),
    (("fc3", 115, 32, 8, True, False), 8, False),
])
def test_measured_entries(key, bw, one_d):
    cfg = tt._transition_bw_config(*key)
    assert cfg.in0_block_w == bw
    assert isinstance(cfg, tt.ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig) == one_d
    assert cfg.out_subblock_h * cfg.out_subblock_w <= 4


def test_fused_silu_rides_the_config():
    assert tt._transition_bw_config("fc1", 115, 8, 32, False, True).fused_activation is not None
    assert tt._transition_bw_config("fc2", 115, 8, 32, False, False).fused_activation is None


def test_bf16_pair_fc3_keeps_ttnns_pick():
    # in0_block_w 4/8 raise its error vs float64 (2.6e-3-4.6e-3 against 1.74e-3), so no entry.
    assert tt._transition_bw_config("fc3", 115, 32, 8, False, False) is None


def test_grid_that_does_not_split_n_declines(monkeypatch):
    monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", (13, 10))
    tt._transition_bw_config.cache_clear()
    assert tt._transition_bw_config("fc2", 115, 8, 32, False, False) is None


def test_lever_is_named_and_in_no_mode_yet():
    assert "transition_bw" in tt.LEVERS
    assert "transition_bw" not in tt.NORMAL_LEVERS | tt.FAST_LEVERS


def test_c384_fc3_names_a_six_column_grid():
    """OpenDDE's c=384 fc3 has 12 output tiles, which 8 columns do not split: its entry asks for 6."""
    cfg = tt._transition_bw_config("fc3", 92, 48, 12, False, False)
    assert cfg.in0_block_w == 16
    assert (cfg.compute_with_storage_grid_size.x, cfg.per_core_N) == (6, 2)
    assert cfg.compute_with_storage_grid_size.y * cfg.per_core_M >= 92
