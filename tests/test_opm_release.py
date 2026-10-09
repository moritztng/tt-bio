"""OuterProductMean frees its operands after a whole contraction, and survives a refused relayout.

The whole path contracts `z = a b^T` once, frees `a` and `b`, then relayouts z. When the first relayout copy is
refused, the call cannot contract again, so it relayouts the same z in row blocks. Both must give the float64
answer, and the whole path must not be slower to reach it.
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

N, C_M, C, C_Z, DEPTH = 64, 64, 32, 128, 2079


def _ref(sd, mh):
    x = torch.nn.functional.layer_norm(mh[0].double(), (C_M,), sd["norm.weight"].double(),
                                       sd["norm.bias"].double(), 1e-5)
    a, b = x @ sd["proj_a.weight"].double().t(), x @ sd["proj_b.weight"].double().t()
    o = torch.einsum("sic,sjd->ijcd", a, b).reshape(N, N, C * C)
    return (o @ sd["proj_o.weight"].double().t() + sd["proj_o.bias"].double()) / mh.shape[1]


@pytest.fixture(autouse=True)
def _clean():
    def reset():
        T._OPM_DRAM_ROW_CAP.clear()
        T.OPM_ROW_STATS.update(whole=0, blocked=0, dram_narrowed=0, join_split=0)
    reset()
    yield
    reset()


@pytest.mark.device
@pytest.mark.parametrize("chunked", [False, True])
def test_refused_relayout_after_release(monkeypatch, chunked):
    dev = T.get_device()
    torch.manual_seed(0)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    sd = {"norm.weight": 1 + 0.1 * torch.randn(C_M), "norm.bias": 0.1 * torch.randn(C_M),
          "proj_a.weight": torch.randn(C, C_M) / C_M ** 0.5,
          "proj_b.weight": torch.randn(C, C_M) / C_M ** 0.5,
          "proj_o.weight": torch.randn(C_Z, C * C) / C,
          "proj_o.bias": 0.1 * torch.randn(C_Z)}
    opm = T.OuterProductMean(sd, ckc, scale_bias=True)
    mh = torch.randn(1, DEPTH, N, C_M)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    m = lambda: [ft(mh[:, s:s + 512]) for s in range(0, DEPTH, 512)] if chunked else ft(mh)
    ref = _ref(sd, mh.bfloat16().float())
    rel = lambda o: float((o - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())

    whole = ttnn.to_torch(opm(m(), None, None)).double().reshape(N, N, C_Z)
    assert T.OPM_ROW_STATS["whole"] == 1 and T.OPM_ROW_STATS["blocked"] == 0

    real, refused = T.ttnn.to_layout, []

    def to_layout(t, layout, *a, **kw):
        # Refuse the whole z's first copy (rows = N * C), once, the way the allocator does.
        if layout == ttnn.ROW_MAJOR_LAYOUT and tuple(t.shape) == (N * C, C * N) and not refused:
            refused.append(1)
            raise RuntimeError("TT_FATAL: Out of Memory: Not enough space to allocate (test)")
        return real(t, layout, *a, **kw)

    monkeypatch.setattr(T.ttnn, "to_layout", to_layout)
    out = ttnn.to_torch(opm(m(), None, None)).double().reshape(N, N, C_Z)
    monkeypatch.setattr(T.ttnn, "to_layout", real)
    assert refused and T.OPM_ROW_STATS["dram_narrowed"] == 1 and T.OPM_ROW_STATS["blocked"] == 1
    assert torch.isfinite(out).all()
    assert rel(whole) < 2e-2 and rel(out) < 1.25 * rel(whole) + 1e-3, (rel(whole), rel(out))
    # The relayout moves numbers; only proj_o's row blocking can round differently.
    assert float((out - whole).abs().max()) < 5e-2
