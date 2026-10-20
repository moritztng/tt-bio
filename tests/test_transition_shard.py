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


def test_lever_is_fast_only_and_rides_on_transition_bw():
    # The shard path runs only beside transition_bw, whose fixed K blocks keep the answer off the core
    # grid (release gate rg:l1-budget). Both are fast-only: normal failed its grade on pose-flip tails.
    assert "transition_shard" in tt.LEVERS
    assert "transition_shard" not in tt.UNGRADED_LEVERS
    assert {"transition_shard", "transition_bw"} <= tt.FAST_LEVERS
    assert not {"transition_shard", "transition_bw"} & tt.NORMAL_LEVERS
    assert "transition_shard" in tt.LATCH_STATS


def test_fc12_interleave_gives_each_core_column_its_fc1_then_fc2_block():
    import torch
    w1, w2 = torch.arange(4 * 16.).reshape(4, 16), -torch.arange(4 * 16.).reshape(4, 16)
    w12 = tt._fc12_interleave(w1, w2, gx=8)
    assert w12.shape == (4, 32)
    for j in range(8):
        assert torch.equal(w12[:, 4 * j:4 * j + 2], w1[:, 2 * j:2 * j + 2])
        assert torch.equal(w12[:, 4 * j + 2:4 * j + 4], w2[:, 2 * j:2 * j + 2])


def test_fc12_interleave_takes_three_weights_for_the_dit_pass():
    import torch
    ws = [k * torch.arange(4 * 48.).reshape(4, 48) for k in (1, -1, 2)]
    w = tt._fc12_interleave(*ws, gx=8)
    assert w.shape == (4, 144)
    for j in range(8):
        for i, wi in enumerate(ws):
            assert torch.equal(w[:, 18 * j + 6 * i:18 * j + 6 * i + 6], wi[:, 6 * j:6 * j + 6])


def test_dit_fc123g_is_ungraded_and_off():
    assert "dit_fc123g" in tt.LEVERS and "dit_fc123g" in tt.LATCH_STATS
    assert "dit_fc123g" not in tt.NORMAL_LEVERS | tt.FAST_LEVERS


def test_fc12g_is_fast_only():
    assert "swiglu_fc12g" in tt.FAST_LEVERS and "silu_f32" in tt.FAST_LEVERS
    assert "swiglu_fc12g" not in tt.NORMAL_LEVERS | tt.UNGRADED_LEVERS


def test_a_shape_fc123g_cannot_build_keeps_dit_shard(monkeypatch):
    """5 samples (pm 10) clashed in L1 inside fc12g and the refusal switched dit_shard off for the whole fold (gfc)."""
    class A:
        padded_shape, dtype = (1, 1, 80 * 32, 768), "bf16"

    class W:
        def __init__(self, n):
            self.shape = (768, n)

    calls = []

    def fc12g(*a, **k):
        raise RuntimeError("Statically allocated circular buffers clash with L1 buffers")

    monkeypatch.setattr(tt, "_fc12g", fc12g)
    monkeypatch.setattr(tt, "_shard_mm_cfg", lambda *a: None)
    monkeypatch.setattr(tt.ttnn, "linear", lambda *a, **k: calls.append("linear") or object())
    monkeypatch.setattr(tt.ttnn, "multiply_", lambda *a, **k: object())
    monkeypatch.setattr(tt.ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(tt, "_TRANSITION_SHARD_REFUSED", set())
    monkeypatch.setattr(tt, "_DIT_FC123G_REFUSED", set())
    out = tt._dit_transition_sharded(A(), W(1536), W(1536), W(1536), W(768), None, w3=object())
    assert out is not None and calls == ["linear"] * 4
    assert not tt._TRANSITION_SHARD_REFUSED and len(tt._DIT_FC123G_REFUSED) == 1


def test_adaln_l1_only_moves_a_windowed_norm_to_l1(monkeypatch):
    """adaln_l1 is placement only, and only where addcmul_at reads the norm back (the windowed s-terms)."""
    seen = []

    class A:
        padded_shape, dtype = (1, 1, 512, 768), tt.ttnn.float32

        def memory_config(self):
            return tt.ttnn.DRAM_MEMORY_CONFIG

    monkeypatch.setattr(tt.ttnn, "layer_norm", lambda a, **k: seen.append(k.get("memory_config")) or a)
    monkeypatch.setattr(tt, "lever", lambda n: n == "adaln_l1")
    tt._adaln_norm(A(), None, windowed=True)
    tt._adaln_norm(A(), None, windowed=False)
    A.padded_shape = (5, 1, 512, 768)   # 7.5 MB of fp32: stays in DRAM
    tt._adaln_norm(A(), None, windowed=True)
    assert seen == [tt.ttnn.L1_MEMORY_CONFIG, None, None]
    assert "adaln_l1" in tt.UNGRADED_LEVERS
