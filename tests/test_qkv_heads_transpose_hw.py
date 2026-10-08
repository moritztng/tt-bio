"""`nlp_create_qkv_heads(..., transpose_k_heads=True)` gives exactly `permute(k, (0, 1, 3, 2))` of the
untransposed split, in fp32 at the Protenix-v2 DiT shape (16 heads of 48 padded to 64, 730 tokens, 5
samples). The fp32 raw-matmul DiT attention relies on it (`AttentionPairBias.__call__`).

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest -s tests/test_qkv_heads_transpose_hw.py
"""
import pytest
import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor

pytestmark = pytest.mark.device


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_transposed_k_equals_permuted_k(dtype):
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    H, d, S, B = 16, 64, 730, 5
    x = torch.randn(B, 1, S, 3 * H * d, generator=torch.Generator().manual_seed(0))
    qkv = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)
    _, k, _ = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False)
    _, kt, _ = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=True)
    want = ttnn.permute(k, (0, 1, 3, 2))
    assert tuple(kt.shape) == tuple(want.shape)
    assert torch.equal(ttnn.to_torch(kt).float(), ttnn.to_torch(want).float())
