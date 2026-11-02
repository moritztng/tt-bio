"""A sampled recycling cycle runs at one depth per fold, the bucket of the fold's deepest draw,
with the rows past its own draw masked. The only reduction over MSA rows is OuterProductMean, so
the padded, masked OPM must land as close to a float64 reference of the UNPADDED draw as the
unpadded OPM does, in each form the trunk hands it: one device tensor, device depth chunks, and
host-parked chunks (a deep alignment's streamed m)."""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

N, C_M, C, C_Z = 64, 64, 32, 128


def _ref(sd, mh):
    x = torch.nn.functional.layer_norm(mh[0].double(), (C_M,), sd["norm.weight"].double(),
                                       sd["norm.bias"].double(), 1e-5)
    a, b = x @ sd["proj_a.weight"].double().t(), x @ sd["proj_b.weight"].double().t()
    o = torch.einsum("sic,sjd->ijcd", a, b).reshape(N, N, C * C)
    return (o @ sd["proj_o.weight"].double().t() + sd["proj_o.bias"].double()) / mh.shape[1]


@pytest.mark.device
@pytest.mark.parametrize("form", ["whole", "chunks", "parked"])
@pytest.mark.parametrize("k,bucket", [(700, 2048), (1500, 2048), (2048, 2048)])
def test_masked_bucket_is_the_unpadded_draw(form, k, bucket):
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
    # The padding rows are not zeros in the trunk (m = linear(0) + s_m), so they are noise here.
    mh = torch.randn(1, bucket, N, C_M).bfloat16()
    ref = _ref(sd, mh[:, :k].float())
    up = lambda x, d=dev: ttnn.from_torch(x.contiguous(), layout=ttnn.TILE_LAYOUT, device=d, dtype=ttnn.bfloat16)

    def run(m_host, mask, n):
        if form == "whole":
            m = up(m_host)
        else:
            m = [up(m_host[:, s:s + 512], None if form == "parked" else dev)
                 for s in range(0, m_host.shape[1], 512)]
        return ttnn.to_torch(opm(m, mask, n)).double().reshape(N, N, C_Z)

    mask = ttnn.from_torch(T.msa_rows_mask(k, bucket), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    padded = run(mh, mask, k)
    unpadded = run(mh[:, :k], None, None)
    rel = lambda x: float((x - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    assert torch.isfinite(padded).all()
    assert rel(unpadded) < 2e-2, rel(unpadded)
    assert rel(padded) < 1.25 * rel(unpadded) + 1e-3, (rel(padded), rel(unpadded))
