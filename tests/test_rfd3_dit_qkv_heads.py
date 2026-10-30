"""RFD3_DIT_QKV_HEADS: qk_rope_heads.rms_heads_packed (q / k RMSNorm + head split of a packed qkv, one program) is at
least as close to a float64 reference as the rms_norm + concat + nlp_create_qkv_heads chain it replaces, at the DiT's
padded page shape (704 tokens, 16 heads of 64), and v comes through bit-exact."""
import pytest
import torch
import ttnn

from tt_bio import qk_rope_heads as Q
from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device


def _up(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def _heads(x, h):   # [1, L, h * 64] -> [1, h, L, 64]
    return x.reshape(1, x.shape[1], h, 64).permute(0, 2, 1, 3)


@pytest.mark.parametrize("length", [704, 96])
def test_rms_heads_packed_matches_float64(length):
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    ckc = M._default_compute_kernel_config()
    h, d, s = 16, 1024, 0.75   # 48 live columns of every 64: gamma and eps scaled as _padded_head_weights does
    g = torch.Generator().manual_seed(length)
    live = torch.zeros(d, dtype=torch.bool).reshape(h, 64)
    live[:, :48] = True
    live = live.reshape(-1)
    qkv = (torch.randn(1, length, 3 * d, generator=g) * 3).bfloat16()
    qkv[..., :d][..., ~live] = 0
    qkv[..., d:2 * d][..., ~live] = 0
    gq, gk = [(torch.rand(d, generator=g) + 0.5).bfloat16() * live for _ in range(2)]
    eps = 1e-6 * s

    def ref_norm(x, gamma):
        x = x.double()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma.double()

    refs = [_heads(ref_norm(qkv[..., :d], gq), h), _heads(ref_norm(qkv[..., d:2 * d], gk), h),
            _heads(qkv[..., 2 * d:].double(), h)]

    x = _up(qkv, dev)
    got = Q.rms_heads_packed(x, _up(gq, dev), _up(gk, dev), eps)
    assert got is not None
    got = [ttnn.to_torch(t).double() for t in got]

    q = ttnn.rms_norm(ttnn.slice(x, [0, 0, 0], [1, length, d]), weight=_up(gq, dev), epsilon=eps,
                      compute_kernel_config=ckc)
    k = ttnn.rms_norm(ttnn.slice(x, [0, 0, d], [1, length, 2 * d]), weight=_up(gk, dev), epsilon=eps,
                      compute_kernel_config=ckc)
    v = ttnn.slice(x, [0, 0, 2 * d], [1, length, 3 * d])
    cat = ttnn.reshape(ttnn.concat([q, k, v], dim=-1), (1, 1, length, 3 * d))
    base = [ttnn.to_torch(t).double() for t in ttnn.experimental.nlp_create_qkv_heads(
        cat, num_heads=h, num_kv_heads=h, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)]

    for name, r, a, b in zip("qk", refs, got, base):
        e_got, e_base = (a - r).abs().max().item(), (b - r).abs().max().item()
        assert e_got <= e_base + 1e-6, (name, e_got, e_base)
    assert torch.equal(got[2], refs[2])
