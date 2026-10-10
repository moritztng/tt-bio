"""`tt_bio.lnbw`, the one-kernel layer-norm backward, against a float64 reference.

The composed backward (`autograd._layer_norm_bw`) is the control: on the same bf16 operands the
kernel must be at least as close to float64 as it is. Measured on qb1 p150a at the round's pair
shape: 1.81e-3 rel L2 for the kernel, 3.84e-3 for the composed path.
"""
import pytest
import torch

device = pytest.mark.device


def _ref(x, g, gamma, eps):
    x, g = x.double(), g.double()
    gm = 1.0 if gamma is None else gamma.double()
    xc = x - x.mean(-1, keepdim=True)
    rstd = (xc.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    norm, dn = xc * rstd, g * gm
    return (dn - dn.mean(-1, keepdim=True) - norm * (dn * norm).mean(-1, keepdim=True)) * rstd


def _blackhole_only():
    from tt_bio import tenstorrent
    if tenstorrent.is_wormhole():
        pytest.skip("kernel graded on Blackhole only; test_declines_on_wormhole covers Wormhole")


def _case(shape, g_dtype, with_gamma, seed=0):
    import ttnn
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    torch.manual_seed(seed)
    K = shape[-1]
    x = (torch.randn(shape) * 0.7 + 0.3).bfloat16().float()
    g = torch.randn(shape) * 1e-2
    g = g.bfloat16().float() if g_dtype == "bf16" else g
    gamma = (1 + 0.2 * torch.randn(K)).bfloat16().float() if with_gamma else None
    up = lambda t, dt: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=dt, device=dev)  # noqa: E731
    xd = up(x.bfloat16(), ttnn.bfloat16)
    gd = up(g.bfloat16(), ttnn.bfloat16) if g_dtype == "bf16" else up(g, ttnn.float32)
    gmd = up(gamma.reshape(1, K).bfloat16(), ttnn.bfloat16) if with_gamma else None
    return x, g, gamma, xd, gd, gmd


def _rel(dx_dev, ref):
    import ttnn
    got = torch.Tensor(ttnn.to_torch(dx_dev)).double().reshape(ref.shape)
    return float((got - ref).norm() / ref.norm())


@device
@pytest.mark.parametrize("shape", [(1, 288, 288, 128), (2, 288, 256), (64, 64, 128)])
@pytest.mark.parametrize("with_gamma", [True, False])
def test_kernel_is_at_least_as_close_to_float64_as_the_composed_path(shape, with_gamma,
                                                                     monkeypatch):
    import ttnn
    from tt_bio import autograd as ag, lnbw
    eps = 1e-5
    g_dtype = "bf16"
    _blackhole_only()
    x, g, gamma, xd, gd, gmd = _case(shape, g_dtype, with_gamma)
    ref = _ref(x, g, gamma, eps)
    monkeypatch.setattr(lnbw, "FUSED", True)
    assert lnbw.eligible(xd, gd, gmd)
    fused = lnbw.layer_norm_bw(xd, gd, gmd, eps)
    assert fused.dtype == (ttnn.float32 if g_dtype == "fp32" else ttnn.bfloat16)
    monkeypatch.setattr(lnbw, "FUSED", False)
    xt = ag.Tensor(xd, requires_grad=True)
    gt = None if gmd is None else ag.Tensor(gmd, requires_grad=False)
    ag._layer_norm_bw(xt, gt, None, eps, ag.precise_config())(gd)
    r_fused, r_comp = _rel(fused, ref), _rel(xt.grad, ref)
    assert r_fused < 3e-3, (r_fused, r_comp)
    assert r_fused <= r_comp * 1.05, (r_fused, r_comp)


@device
def test_the_composed_backward_routes_through_the_kernel_when_armed(monkeypatch):
    from tt_bio import autograd as ag, lnbw
    _blackhole_only()
    x, g, gamma, xd, gd, gmd = _case((64, 64, 128), "bf16", True)
    monkeypatch.setattr(lnbw, "FUSED", True)
    before = lnbw.REACH["served"]
    xt = ag.Tensor(xd, requires_grad=True)
    ag._layer_norm_bw(xt, ag.Tensor(gmd, requires_grad=False), None, 1e-5, ag.precise_config())(gd)
    assert lnbw.REACH["served"] == before + 1
    # A weight gradient needs the composed path's `norm`, so it never routes here.
    xt = ag.Tensor(xd, requires_grad=True)
    gw = ag.Tensor(gmd, requires_grad=True)
    ag._layer_norm_bw(xt, gw, None, 1e-5, ag.precise_config())(gd)
    assert lnbw.REACH["served"] == before + 1 and gw.grad is not None


@device
def test_declines_a_float32_cotangent(monkeypatch):
    """The composed path is exact float32 there (1.5e-4); the kernel's FPU reads TF32 (1.3e-3)."""
    from tt_bio import lnbw
    monkeypatch.setattr(lnbw, "FUSED", True)
    _, _, _, xd, gd, gmd = _case((64, 64, 128), "fp32", True)
    assert not lnbw.eligible(xd, gd, gmd)


@device
def test_declines_a_width_whose_reciprocal_is_not_exact(monkeypatch):
    import ttnn
    from tt_bio import lnbw
    from tt_bio.tenstorrent import get_device
    monkeypatch.setattr(lnbw, "FUSED", True)
    t = ttnn.from_torch(torch.randn(64, 384).bfloat16(), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=get_device())
    assert not lnbw.eligible(t, t, None)


def test_declines_on_wormhole(monkeypatch):
    """Card-free: the kernel was graded on Blackhole, so a Wormhole chip keeps the composed path."""
    from tt_bio import lnbw, tenstorrent
    monkeypatch.setattr(lnbw, "FUSED", True)
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: True)
    before = lnbw.REACH["declined: arch"]
    assert not lnbw.eligible(None, None, None)
    assert lnbw.REACH["declined: arch"] == before + 1
