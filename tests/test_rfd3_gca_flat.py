"""The gated cross attention's flat inputs give the padded form's answer.

A downcast passes one query per token as [B,T,C] instead of q.unsqueeze(2); an upcast passes its 3 keys a token
as [B,T*3,C] with n_key=3 instead of [B,T,3,C]. Both run the row-wise ops on the real rows only. The answer has
to match the padded form to bf16 rounding (a different row count may pick a different matmul blocking).
"""
import pytest
import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device


def _weights(g, c_query, c_kv, c_model=128):
    r = lambda *s: torch.randn(*s, generator=g) / s[-1] ** 0.5  # noqa: E731
    one = lambda n: 1 + 0.1 * torch.randn(n, generator=g)  # noqa: E731
    return {"ln_q.weight": one(c_query), "ln_kv.weight": one(c_kv), "to_q.weight": r(c_model, c_query),
            "to_k.weight": r(c_model, c_kv), "to_v.weight": r(c_model, c_kv), "to_g.0.weight": r(c_model, c_query),
            "k_norm.weight": one(c_model), "q_norm.weight": one(c_model),
            "to_out.0.weight": r(c_query, c_model), "to_out.0.bias": 0.1 * torch.randn(c_query, generator=g)}


def _rel(a, b):
    a, b = ttnn.to_torch(a).double(), ttnn.to_torch(b).double()
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("T", [70, 96])
def test_downcast_flat_query(T, monkeypatch):
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    g = torch.Generator().manual_seed(T)
    gca = M.GatedCrossAttention(_weights(g, 768, 128), M._default_compute_kernel_config(),
                                c_query=768, c_kv=128, dtype=ttnn.bfloat16)
    up = lambda t: ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa
    a, kv = torch.randn(1, T, 768, generator=g), torch.randn(1, T, 32, 128, generator=g)
    mask = torch.rand(T, 1, 32, generator=g) > 0.3
    mask[..., 0] = True
    monkeypatch.setattr(M, "_GCA_FLAT_Q", False)
    ref = ttnn.squeeze(gca.run_device(ttnn.unsqueeze(up(a), 2), up(kv), mask.unsqueeze(0)), 2)
    monkeypatch.setattr(M, "_GCA_FLAT_Q", True)
    out = gca.run_device(up(a), up(kv), mask.unsqueeze(0))
    assert tuple(out.shape) == (1, T, 768)
    assert _rel(out, ref) < 1e-2


@pytest.mark.parametrize("T", [70, 96])
def test_upcast_flat_keys(T):
    ensure_p300_mesh_descriptor()
    dev = M.get_device()
    g = torch.Generator().manual_seed(T + 1)
    gca = M.GatedCrossAttention(_weights(g, 128, 256), M._default_compute_kernel_config(),
                                c_query=128, c_kv=256, dtype=ttnn.bfloat16)
    up = lambda t: ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa
    q, a = torch.randn(1, T, 32, 128, generator=g), torch.randn(1, T, 768, generator=g)
    ref = gca.run_device(up(q), ttnn.reshape(up(a), (1, T, 3, 256)))
    out = gca.run_device(up(q), ttnn.reshape(up(a), (1, T * 3, 256)), n_key=3)
    assert _rel(out, ref) < 1e-2
