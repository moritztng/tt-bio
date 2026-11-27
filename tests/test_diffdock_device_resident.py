"""The resident conv stack builds the edge input on the card from split FC weights; card-free."""

import pytest
import torch

from tt_bio.diffdock.conv import TensorProductConvLayer
from tt_bio.diffdock.device import GraphCache, fc1_blocks, pad32

NS = 48


def _pad(x, w):
    return torch.nn.functional.pad(x, (0, w - x.shape[1]))


@pytest.mark.parametrize("in_irreps", ["48x0e", "48x0e + 10x1o + 10x1e + 10x0o"])
def test_split_fc1_equals_concatenated_edge_input(in_irreps):
    torch.manual_seed(0)
    layer = TensorProductConvLayer(in_irreps, "1x0e + 1x1o + 1x2e", "48x0e + 10x1o", 3 * NS,
                                   hidden_features=3 * NS).double().eval()
    for p in layer.fc.parameters():
        torch.nn.init.normal_(p, std=0.3)
    fc = layer.fc
    N, E, in_dim = 37, 90, layer.tp.in_irreps.dim
    in_pad = pad32(in_dim)
    x = torch.randn(N, in_dim, dtype=torch.float64)
    ea = torch.randn(E, NS, dtype=torch.float64)
    src, dst = torch.randint(0, N, (2, E))
    ref = torch.relu(fc[0](torch.cat([ea, x[src, :NS], x[dst, :NS]], 1)))
    w1a, w1s, w1d, b1 = fc1_blocks(fc, NS, in_pad)
    xp = _pad(x, in_pad)
    got = torch.relu(_pad(ea, w1a.shape[0]) @ w1a + (xp @ w1s)[src] + (xp @ w1d)[dst] + b1)
    assert got.shape[1] == pad32(3 * NS)
    hidden = ref.shape[1]
    assert (got[:, :hidden] - ref).abs().max() <= 1e-12 * ref.abs().max()
    assert not got[:, hidden:].any()


def test_graph_cache_reuses_rebuilds_and_evicts():
    pytest.importorskip("ttnn")
    built = []
    cache = GraphCache(size=2)

    def build(n):
        return lambda: built.append(n) or dict(n_out=32, E=n, tag=n)

    ei = {n: torch.arange(2 * n).reshape(2, n) for n in (32, 64, 96)}
    assert cache.get(32, ei[32], build(32))["tag"] == 32
    cache.get(32, ei[32].clone(), build(32))                  # same content: no rebuild
    cache.get(32, ei[32].flip(1), build(32))                  # same key, new content: rebuild
    assert built == [32, 32]
    cache.get(64, ei[64], build(64))
    cache.get(32, ei[32].flip(1), build(32))                  # 32 most recent
    cache.get(96, ei[96], build(96))                          # evicts 64, not 32
    assert list(cache.entries) == [32, 96] and built == [32, 32, 64, 96]
    assert cache.nbytes() == sum(32 * n * 2 + n * 4 for n in (32, 96))
