"""OuterProductMean pads its contraction depth with zero rows (`TT_BIO_OPM_KPAD`) and still
averages over the real depth: padded and unpadded land equally close to a float64 reference."""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

N, C_M, C, C_Z = 64, 64, 32, 128


@pytest.mark.parametrize("depth,rows", [(9947, 37), (9984, 0), (12830, 34), (4097, 31), (5889, 63), (13602, 30)])
def test_pad_rows(depth, rows):
    assert T.opm_kpad_rows(depth) == rows
    assert (depth + rows) % (32 * T.OPM_K_BLOCK) == 0


@pytest.mark.parametrize("depth", [80, 300])
def test_shallow_is_not_padded(depth):
    assert T.opm_kpad_rows(depth) == 0


def test_off(monkeypatch):
    monkeypatch.setattr(T, "_OPM_KPAD", False)
    assert T.opm_kpad_rows(9947) == 0


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


@pytest.mark.device
@pytest.mark.parametrize("j,s", [(736, 9984), (736, 9947), (64, 97), (40, 97)])
def test_flat_b_is_the_row_major_reshape(j, s):
    """`opm_flat_b`'s tile view holds exactly the values of the row-major (D*J, S) reshape."""
    dev = T.get_device()
    torch.manual_seed(0)
    b = torch.randn(32, j, s).bfloat16()
    t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = T.opm_flat_b(t)
    assert tuple(out.shape) == (32 * j, s) and out.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(out), b.reshape(32 * j, s))


class _Grid:
    x, y = 8, 9


@pytest.mark.parametrize("tokens,kt,pcm,obw", [(736, 312, 84, 23), (512, 312, 60, 16), (1024, 432, 116, 16)])
def test_contract_config_picks_the_measured_plan(tokens, kt, pcm, obw):
    cfg = T.opm_contract_config(tokens, tokens, kt, _Grid)
    assert (cfg.per_core_M, cfg.in0_block_w, cfg.out_block_h, cfg.out_block_w) == (pcm, T.OPM_K_BLOCK, 4, obw)
    assert cfg.out_subblock_h * cfg.out_subblock_w == 4


def test_contract_config_never_blocks_k_wider_than_the_clean_width():
    """4+ K tiles per block put single elements off by 1/2/4 on Wormhole (see OPM_K_BLOCK)."""
    for kt in range(2, 700):
        cfg = T.opm_contract_config(736, 736, kt, _Grid)
        assert cfg is None or 1 < cfg.in0_block_w <= T.OPM_K_BLOCK, kt


def test_contract_config_small_prime_or_off(monkeypatch):
    assert T.opm_contract_config(32, 32, 312, _Grid) is None
    assert T.opm_contract_config(736, 736, 311, _Grid) is None   # one-tile blocks: ttnn's plan is faster
    monkeypatch.setattr(T, "_OPM_CFG", False)
    assert T.opm_contract_config(736, 736, 312, _Grid) is None


@pytest.mark.device
@pytest.mark.parametrize("m,n,k", [(3072, 2048, 1056), (2944, 2944, 768)])
def test_contract_config_is_as_close_to_float64(m, n, k):
    """The rounded-up per_core_M leaves the last core row partly empty; every row still lands."""
    dev = T.get_device()
    torch.manual_seed(0)
    a, b = (torch.randn(m, k) / k ** 0.5).bfloat16(), torch.randn(n, k).bfloat16()
    ref = a.double() @ b.double().T
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg = T.opm_contract_config(m // 32, n // 32, k // 32, dev.compute_with_storage_grid_size())
    assert cfg is not None
    err = {}
    for name, pc in (("auto", None), ("cfg", cfg)):
        out = ttnn.to_torch(ttnn.matmul(ft(a), ft(b), transpose_b=True, program_config=pc,
                                        compute_kernel_config=ckc)).double()
        err[name] = float((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
        assert (out - ref).abs().max() < 0.25, name   # outputs ~N(0,1): no element off by a whole unit
    assert err["cfg"] < 1.1 * err["auto"] + 1e-4, err
