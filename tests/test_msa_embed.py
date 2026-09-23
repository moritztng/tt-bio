"""`msa_embed` in depth chunks is the whole-depth embedding, bit for bit.

Past 1 GiB of alignment feature the Protenix/OpenDDE and OpenFold3 trunks upload and project
the MSA one depth chunk at a time and keep `m` on the host. The projection is a linear over
channels plus the single representation broadcast over depth, so each chunk must come back
holding exactly the rows the whole pass computes.
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device


@pytest.fixture(scope="module")
def dev():
    return T.get_device()


def _project(dev):
    torch.manual_seed(0)
    n, c, c_m = 64, 34, 64
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    w = ft(torch.randn(c, c_m) / c ** 0.5)
    s = ft(torch.randn(1, 1, n, c_m))
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    return n, c, lambda x: ttnn.add(ttnn.linear(x, w, compute_kernel_config=ckc), s)


@pytest.mark.parametrize("rows", [32, 48])
def test_depth_chunks_are_the_whole_pass(dev, monkeypatch, rows):
    n, c, project = _project(dev)
    feat = torch.randn(1, 80, n, c)             # 80 rows: whole chunks plus a ragged tail
    feat[..., :32] = (feat[..., :32] > 1).float()
    monkeypatch.setenv("TT_BIO_MSA_HOST_OFFLOAD_MIN_BYTES", str(1 << 40))
    whole = T.msa_embed(feat, project, rows)
    assert not torch.is_tensor(whole)
    whole = ttnn.to_torch(whole)
    monkeypatch.setenv("TT_BIO_MSA_HOST_OFFLOAD_MIN_BYTES", "0")
    chunked = T.msa_embed(feat, project, rows)
    assert torch.is_tensor(chunked) and tuple(chunked.shape) == (1, 80, n, 64)
    assert torch.equal(chunked, whole)


def test_a_device_feature_is_embedded_whole_and_consumed(dev, monkeypatch):
    n, c, project = _project(dev)
    feat = torch.randn(1, 40, n, c)
    monkeypatch.setenv("TT_BIO_MSA_HOST_OFFLOAD_MIN_BYTES", str(1 << 40))
    ref = ttnn.to_torch(T.msa_embed(feat, project))
    x = ttnn.from_torch(feat, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = T.msa_embed(x, project)
    assert not x.is_allocated()
    assert torch.equal(ttnn.to_torch(out), ref)
