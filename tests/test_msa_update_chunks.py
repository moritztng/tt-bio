"""`msa_update_chunks` is the whole-depth MSA update, bit for bit, and PWA reads `z` once for it.

PairWeightedAveraging's token weights are a function of the pair alone, so the chunked update
computes them once (`head_weights`) and hands them to every depth chunk instead of norming the
whole pair again per chunk. At 1536 tokens and c_z=384 that normed pair, and the clone of `z`
the loop used to take, were each a 1811939328 B buffer a Wormhole chip refused.
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device

N, C_M, C_Z, HD, NH = 64, 64, 128, 8, 8


@pytest.fixture(scope="module")
def dev():
    return T.get_device()


def _setup(dev):
    torch.manual_seed(0)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    sd = {"norm_m.weight": 1 + 0.1 * torch.randn(C_M), "norm_m.bias": 0.1 * torch.randn(C_M),
          "norm_z.weight": 1 + 0.1 * torch.randn(C_Z), "norm_z.bias": 0.1 * torch.randn(C_Z),
          "proj_m.weight": torch.randn(NH * HD, C_M) / C_M ** 0.5,
          "proj_g.weight": torch.randn(NH * HD, C_M) / C_M ** 0.5,
          "proj_z.weight": torch.randn(NH, C_Z) / C_Z ** 0.5,
          "proj_o.weight": torch.randn(C_M, NH * HD) / (NH * HD) ** 0.5}
    pwa = T.PairWeightedAveraging(HD, NH, sd, ckc)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    wt = ft(torch.randn(C_M, C_M) / C_M ** 0.5)
    transition = lambda x: ttnn.linear(x, wt, compute_kernel_config=ckc)
    mask = torch.zeros(1, N)
    mask[:, -5:] = -1e4                             # the trunk's additive token mask
    return pwa, transition, ft, ft(mask)


def test_head_weights_are_the_calls_own(dev):
    pwa, _, ft, mask = _setup(dev)
    m, z = ft(torch.randn(1, 40, N, C_M)), ft(torch.randn(1, N, N, C_Z))
    ws = pwa.head_weights(z, mask)
    assert len(ws) == NH
    assert torch.equal(ttnn.to_torch(pwa(m, None, weights=ws)), ttnn.to_torch(pwa(m, z, mask)))
    assert all(w.is_allocated() for w in ws)        # the caller's to free


@pytest.mark.parametrize("rows", [32, 48])
def test_chunked_update_is_the_whole_update(dev, rows):
    pwa, transition, ft, mask = _setup(dev)
    mh, zh = torch.randn(1, 80, N, C_M), torch.randn(1, N, N, C_Z)   # 80 rows: ragged tail
    m, z = ft(mh), ft(zh)
    t1 = ttnn.add(m, ttnn.reshape(pwa(m, z, mask), tuple(m.shape)))
    whole = ttnn.to_torch(ttnn.add(t1, ttnn.reshape(transition(t1), tuple(t1.shape))))
    parts = T.msa_update_chunks(T.msa_depth_chunks(mh, rows), z, pwa, transition, mask)
    assert len(parts) == -(-80 // rows)
    assert torch.equal(torch.cat([ttnn.to_torch(p) for p in parts], dim=1), whole)
    assert z.is_allocated() and torch.equal(ttnn.to_torch(z), ttnn.to_torch(ft(zh)))
