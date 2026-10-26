"""The pairformer attention through the fused SDPA (RFD3_PF_SDPA) is at least as close to a float64 reference as the
manual chain it replaces, with the pair bias from the linear and from dit_pair_bias, at a whole and a partial tile."""
import pytest
import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor
from tt_bio.rfd3 import model as M

pytestmark = pytest.mark.device
CS, CZ, H = 384, 128, 16
HD = CS // H


def _weights(g):
    r = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).bfloat16().float()  # noqa: E731
    w = {f"to_{k}.weight": r(CS, CS, sc=CS ** -0.5) for k in ("q", "k", "v", "a")}
    w["to_g.0.weight"] = r(CS, CS, sc=CS ** -0.5)
    w["to_q.bias"] = r(CS, sc=0.1)
    w["to_b.weight"] = r(H, CZ, sc=CZ ** -0.5)
    for k in ("ln_1.weight", "to_k.ln.weight", "to_v.ln.weight"):
        w[k] = 1 + r(CS, sc=0.1)
    w["ln_0.weight"] = 1 + r(CZ, sc=0.1)
    return {k: v.bfloat16().float() for k, v in w.items()}


def _reference(w, s, z):
    d = {k: v.double() for k, v in w.items()}
    rms = lambda x, g: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g  # noqa: E731
    a = rms(s.double(), d["ln_1.weight"])
    q = a @ d["to_q.weight"].t() + d["to_q.bias"]
    k = rms(a @ d["to_k.weight"].t(), d["to_k.ln.weight"])
    v = rms(a @ d["to_v.weight"].t(), d["to_v.ln.weight"])
    b = rms(z.double(), d["ln_0.weight"]) @ d["to_b.weight"].t()
    heads = lambda x: x.reshape(1, -1, H, HD).transpose(1, 2)              # noqa: E731
    sc = heads(q) @ heads(k).transpose(-1, -2) / HD ** 0.5 + b.permute(0, 3, 1, 2)
    o = (sc.softmax(-1) @ heads(v)).transpose(1, 2).reshape(1, -1, CS)
    return (o * torch.sigmoid(a @ d["to_g.0.weight"].t())) @ d["to_a.weight"].t()


@pytest.mark.parametrize("I", [96, 70])
@pytest.mark.parametrize("dpb", [False, True])
def test_pf_sdpa_matches_float64(I, dpb, monkeypatch):
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    g = torch.Generator().manual_seed(I)
    w = _weights(g)
    s_t = torch.randn(1, I, CS, generator=g).bfloat16()
    z_t = torch.randn(1, I, I, CZ, generator=g).bfloat16()
    ref = _reference(w, s_t, z_t)
    attn = M.PairformerAttention(w, M._default_compute_kernel_config(), dtype=ttnn.bfloat16)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    monkeypatch.setattr(M, "_PF_DPB", dpb)
    err = {}
    for sdpa in (False, True):
        monkeypatch.setattr(M, "_PF_SDPA", sdpa)
        out = ttnn.to_torch(attn(up(s_t), up(z_t))).double()[:, :I, :CS]
        assert torch.isfinite(out).all()
        err[sdpa] = float((out - ref).norm() / ref.norm())
    assert err[True] <= 1.25 * err[False] + 1e-3, err
