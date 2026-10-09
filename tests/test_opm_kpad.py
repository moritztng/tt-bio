"""OuterProductMean pads its contraction depth with zero rows (`TT_BIO_OPM_KPAD_TILES`) and still
averages over the real depth: padded and unpadded land equally close to a float64 reference."""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

N, C_M, C, C_Z = 64, 64, 32, 128


@pytest.mark.parametrize("depth,tiles,rows", [(9947, 2, 37), (9984, 2, 0), (80, 2, 48), (80, 1, 0),
                                              (80, 4, 48), (130, 4, 126)])
def test_pad_rows(monkeypatch, depth, tiles, rows):
    monkeypatch.setattr(T, "_OPM_KPAD_TILES", tiles)
    assert T.opm_kpad_rows(depth) == rows


def _ref(sd, mh):
    x = torch.nn.functional.layer_norm(mh[0].double(), (C_M,), sd["norm.weight"].double(),
                                       sd["norm.bias"].double(), 1e-5)
    a, b = x @ sd["proj_a.weight"].double().t(), x @ sd["proj_b.weight"].double().t()
    o = torch.einsum("sic,sjd->ijcd", a, b).reshape(N, N, C * C)
    return (o @ sd["proj_o.weight"].double().t() + sd["proj_o.bias"].double()) / mh.shape[1]


@pytest.mark.device
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("depth", [80, 97])
def test_padded_is_as_close_to_float64(monkeypatch, chunked, depth):
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
    mh = torch.randn(1, depth, N, C_M)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ref = _ref(sd, mh.bfloat16().float())
    err = {}
    for tiles in (1, 2):
        monkeypatch.setattr(T, "_OPM_KPAD_TILES", tiles)
        m = [ft(mh[:, s:s + 32]) for s in range(0, depth, 32)] if chunked else ft(mh)
        out = ttnn.to_torch(opm(m, None, None)).double().reshape(N, N, C_Z)
        assert torch.isfinite(out).all()
        err[tiles] = float((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    assert err[1] < 2e-2, err
    assert err[2] < 1.25 * err[1] + 1e-3, err
