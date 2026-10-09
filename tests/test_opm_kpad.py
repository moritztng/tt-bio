"""OuterProductMean pads its contraction depth with zero rows (`TT_BIO_OPM_KPAD`) and still
averages over the real depth: padded and unpadded land equally close to a float64 reference."""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

N, C_M, C, C_Z = 64, 64, 32, 128


@pytest.mark.parametrize("depth,grid_x,rows", [(9947, 8, 37), (9984, 8, 0), (12830, 8, 226), (4097, 8, 255),
                                               (9947, 13, 37), (5889, 13, 351)])
def test_pad_rows(depth, grid_x, rows):
    assert T.opm_kpad_rows(depth, grid_x) == rows
    assert (depth + rows) % (32 * grid_x) == 0


@pytest.mark.parametrize("depth,grid_x", [(80, 8), (1100, 8), (1100, 13)])
def test_shallow_is_not_padded(depth, grid_x):
    assert T.opm_kpad_rows(depth, grid_x) == 0


def test_off(monkeypatch):
    monkeypatch.setattr(T, "_OPM_KPAD", False)
    assert T.opm_kpad_rows(9947, 8) == 0


def _ref(sd, mh):
    x = torch.nn.functional.layer_norm(mh[0].double(), (C_M,), sd["norm.weight"].double(),
                                       sd["norm.bias"].double(), 1e-5)
    a, b = x @ sd["proj_a.weight"].double().t(), x @ sd["proj_b.weight"].double().t()
    o = torch.einsum("sic,sjd->ijcd", a, b).reshape(N, N, C * C)
    return (o @ sd["proj_o.weight"].double().t() + sd["proj_o.bias"].double()) / mh.shape[1]


@pytest.mark.device
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("depth", [2400, 2079])
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
    for on in (False, True):
        monkeypatch.setattr(T, "_OPM_KPAD", on)
        m = [ft(mh[:, s:s + 512]) for s in range(0, depth, 512)] if chunked else ft(mh)
        out = ttnn.to_torch(opm(m, None, None)).double().reshape(N, N, C_Z)
        assert torch.isfinite(out).all()
        err[on] = float((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    assert err[False] < 2e-2, err
    assert err[True] < 1.25 * err[False] + 1e-3, err
