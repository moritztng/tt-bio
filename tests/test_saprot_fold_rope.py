"""SaProt-35M folds RoPE into its q/k/v weights (head_dim 24 padded to 32). Host-only check
that the folded projection, tables and padded output projection reproduce the reference
ESM-2 attention in float64: no device, no checkpoint."""
import pytest
import torch

saprot = pytest.importorskip("tt_bio.saprot")


def _reference(x, w, b, ow, ob, H, d):
    B, L, _ = x.shape
    q, k, v = (x @ w.T + b).view(B, L, 3, H, d).permute(2, 0, 3, 1, 4)
    inv = 1.0 / (saprot.ROPE_BASE ** (torch.arange(0, d, 2, dtype=x.dtype) / d))
    f = torch.outer(torch.arange(L, dtype=x.dtype), inv)
    cos, sin = torch.cat([f.cos(), f.cos()], -1), torch.cat([f.sin(), f.sin()], -1)
    rh = lambda t: torch.cat([-t[..., d // 2:], t[..., : d // 2]], -1)
    q, k = q * cos + rh(q) * sin, k * cos + rh(k) * sin
    o = torch.softmax(q @ k.transpose(-1, -2) * d ** -0.5, -1) @ v
    return o.transpose(1, 2).reshape(B, L, H * d) @ ow.T + ob


def _folded(x, w, b, ow, ob, H, d):
    B, L, _ = x.shape
    wf, bf, wr, br = saprot._fold_rope_qkv(w, b, H, d)
    cos, sin = saprot._fold_rope_tables_torch(L, H, d, x.dtype)
    qkv = (x @ wf.T + bf) * cos + (x @ wr.T + br) * sin
    q, k, v = qkv.view(B, L, 3, H, 32).permute(2, 0, 3, 1, 4)
    o = torch.softmax(q @ k.transpose(-1, -2) * d ** -0.5, -1) @ v
    return o.transpose(1, 2).reshape(B, L, H * 32) @ saprot._pad_heads_in(ow, H, d).T + ob


def test_fold_rope_matches_reference_float64():
    torch.manual_seed(0)
    H, d, B, L = 20, 24, 2, 70
    hid = H * d
    x = torch.randn(B, L, hid, dtype=torch.float64)
    w, b = torch.randn(3 * hid, hid, dtype=torch.float64) / hid ** 0.5, torch.randn(3 * hid, dtype=torch.float64)
    ow, ob = torch.randn(hid, hid, dtype=torch.float64) / hid ** 0.5, torch.randn(hid, dtype=torch.float64)
    ref, got = _reference(x, w, b, ow, ob, H, d), _folded(x, w, b, ow, ob, H, d)
    assert torch.isfinite(got).all()
    assert (ref - got).abs().max().item() < 1e-10


def test_rot_weight_is_a_signed_permutation():
    torch.manual_seed(1)
    H, d = 20, 24
    w, b = torch.randn(3 * H * d, H * d), torch.randn(3 * H * d)
    wf, bf, wr, br = saprot._fold_rope_qkv(w, b, H, d)
    assert wf.shape == wr.shape == (3 * H * 32, H * d)
    # bit-exact: every rotated row is a sign-flipped copy of a padded row, v rows are zero
    assert set(wr.abs().reshape(3, H, 32, -1)[:2].flatten().tolist()) <= set(wf.abs().flatten().tolist())
    assert not wr.reshape(3, -1)[2].any() and not br.reshape(3, -1)[2].any()
    assert not wf.reshape(3, H, 32, -1)[:, :, d:].any()


def _paired(x, w, b, ow, ob, H, d):
    """The pairs=True path as rotary_embedding_llama computes it: x * cos + (x @ T) * sin on 32-wide heads."""
    B, L, _ = x.shape
    wf, bf, _, _ = saprot._fold_rope_qkv(w, b, H, d, pairs=True)
    cos, sin = (t.to(x.dtype) for t in saprot.pair_rope_tables_torch(L, d, torch.float64))
    q, k, v = (x @ wf.T + bf).view(B, L, 3, H, 32).permute(2, 0, 3, 1, 4)
    t = saprot._pair_rotation().to(x.dtype)
    q, k = q * cos + (q @ t) * sin, k * cos + (k @ t) * sin
    o = torch.softmax(q @ k.transpose(-1, -2) * d ** -0.5, -1) @ v
    return o.transpose(1, 2).reshape(B, L, H * 32) @ saprot._pad_heads_in(ow, H, d).T + ob


def test_pair_rope_matches_reference_float64():
    torch.manual_seed(2)
    H, d, B, L = 20, 24, 2, 70
    hid = H * d
    x = torch.randn(B, L, hid, dtype=torch.float64)
    w, b = torch.randn(3 * hid, hid, dtype=torch.float64) / hid ** 0.5, torch.randn(3 * hid, dtype=torch.float64)
    ow, ob = torch.randn(hid, hid, dtype=torch.float64) / hid ** 0.5, torch.randn(hid, dtype=torch.float64)
    ref, got = _reference(x, w, b, ow, ob, H, d), _paired(x, w, b, ow, ob, H, d)
    assert (ref - got).abs().max().item() < 1e-10
