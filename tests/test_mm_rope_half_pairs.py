"""mm_rope's half-pair layout for heads narrower than 32 (SaProt 35M, head_dim 24): the permuted projection, the
epilogue's tile-pair RoPE and the writer's face split (mirrored by mm_rope.half_pair_heads) give NeoX RoPE q, k and the
padded v, against float64. CPU only."""
import pytest
import torch

pytest.importorskip("ttnn")
from tt_bio import mm_rope  # noqa: E402


def _neox(x, n_heads, d, seq_len, base=10000.0):
    """[L, H d] -> [H, L, d] NeoX RoPE, as esm's RotaryEmbedding (interleaved=False)."""
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=x.dtype) / d))
    f = torch.outer(torch.arange(seq_len, dtype=x.dtype), inv)
    cos, sin = torch.cat((f.cos(), f.cos()), -1), torch.cat((f.sin(), f.sin()), -1)
    x = x.unflatten(-1, (n_heads, d)).movedim(-2, -3)
    rot = torch.cat((-x[..., d // 2:], x[..., :d // 2]), -1)
    return x * cos + rot * sin


def _as_head_tile(x, d):
    """[.., d] NeoX head -> its [.., 32] half-pair head tile [x1 0.. | x2 0..]."""
    pad = lambda t: torch.nn.functional.pad(t, (0, 16 - d // 2))
    return torch.cat((pad(x[..., :d // 2]), pad(x[..., d // 2:])), -1)


@pytest.mark.parametrize("n_heads,d", [(20, 24), (4, 32), (2, 8)])
def test_half_pairs_match_neox(n_heads, d):
    torch.manual_seed(0)
    hidden, seq_len = 48, 64
    w = torch.randn(3 * n_heads * d, hidden, dtype=torch.float64)
    b = torch.randn(3 * n_heads * d, dtype=torch.float64)
    x = torch.randn(seq_len, hidden, dtype=torch.float64)
    wp, bp = mm_rope.half_pairs(w, b, n_heads, d)
    assert wp.shape == (hidden, 3 * n_heads * 32) and bp.shape == (1, 3 * n_heads * 32)
    cos, sin = mm_rope.half_pair_tables(seq_len, d, dtype=torch.float64)
    assert torch.equal(cos[:, :32], cos[:, 32:]) and torch.equal(sin[:, :32], sin[:, 32:])
    q, k, v = mm_rope.half_pair_heads(x @ wp + bp, cos, sin, n_heads)
    ref = (x @ w.t() + b).unflatten(-1, (3, n_heads * d))
    rq, rk = (_neox(ref[:, j], n_heads, d, seq_len) for j in range(2))
    torch.testing.assert_close(q, _as_head_tile(rq, d), rtol=0, atol=1e-10)
    torch.testing.assert_close(k, _as_head_tile(rk, d), rtol=0, atol=1e-10)
    rv = ref[:, 2].unflatten(-1, (n_heads, d)).movedim(-2, -3)
    torch.testing.assert_close(v, torch.nn.functional.pad(rv, (0, 32 - d)), rtol=0, atol=1e-10)
    # what SDPA consumes: q . k over the 32-column tiles equals the 24-dim dot product
    torch.testing.assert_close(q @ k.transpose(-1, -2), rq @ rk.transpose(-1, -2), rtol=1e-12, atol=1e-9)


def test_half_pair_layout_on_a_five_wide_in1_axis():
    # 35M: 60 tiles = 20 q/k tile pairs + 20 v tiles. Whole pairs need 5 in1 cores (8 a core = 4 pairs, 4 v); 8 split them.
    assert mm_rope.layout(60, 5, 4) == (12, 8, 40)
    assert mm_rope.layout(60, 8, 4) is None
