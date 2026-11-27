"""The resident conv stack builds the edge input on the card from split FC weights; card-free."""

import pytest
import torch

from tt_bio.diffdock.conv import TensorProductConvLayer
from tt_bio.diffdock.device import fc1_blocks, pad32
from tt_bio.diffdock.edge_tile import EdgeTilePlan

NS = 48


def _pad(x, w):
    return torch.nn.functional.pad(x, (0, w - x.shape[1]))


@pytest.mark.parametrize("form", ["gemm", "tile"])
@pytest.mark.parametrize("in_irreps", ["48x0e", "48x0e + 10x1o + 10x1e + 10x0o"])
def test_split_fc1_equals_concatenated_edge_input(form, in_irreps):
    torch.manual_seed(0)
    layer = TensorProductConvLayer(in_irreps, "1x0e + 1x1o + 1x2e", "48x0e + 10x1o", 3 * NS,
                                   hidden_features=3 * NS).double().eval()
    for p in layer.fc.parameters():
        torch.nn.init.normal_(p, std=0.3)
    fc = layer.fc
    plan = EdgeTilePlan(layer.tp, fc) if form == "tile" else None
    N, E, in_dim = 37, 90, layer.tp.in_irreps.dim
    in_pad = pad32(in_dim)
    x = torch.randn(N, in_dim, dtype=torch.float64)
    ea = torch.randn(E, NS, dtype=torch.float64)
    src, dst = torch.randint(0, N, (2, E))
    e = torch.cat([ea, x[src, :NS], x[dst, :NS]], 1)
    if plan is None:
        ref = torch.relu(fc[0](e))
    else:
        ref = torch.relu(_pad(e, plan.feat_pad) @ plan.W1 + plan.b1)
    w1a, w1s, w1d, b1 = fc1_blocks(fc, NS, in_pad, plan)
    xp = _pad(x, in_pad)
    got = torch.relu(_pad(ea, w1a.shape[0]) @ w1a + (xp @ w1s)[src] + (xp @ w1d)[dst] + b1)
    assert got.shape[1] == (plan.hidden_pad if plan is not None else pad32(3 * NS))
    hidden = ref.shape[1]
    assert (got[:, :hidden] - ref).abs().max() <= 1e-12 * ref.abs().max()
    assert not got[:, hidden:].any()
