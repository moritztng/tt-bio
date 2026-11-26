"""The fused edge-tile schedule against the host tensor product, in float64, card-free."""

import pytest
import torch

from tt_bio.diffdock.conv import TensorProductConvLayer
from tt_bio.diffdock.edge_tile import EdgeTilePlan, emulate

SIGS = [
    ("48x0e", "48x0e + 10x1o", 41),
    ("48x0e + 10x1o + 10x1e + 10x0o", "48x0e + 10x1o + 10x1e + 10x0o", 144),
    ("48x0e + 10x1o + 10x1e + 10x0o", "2x1o + 2x1e", 99),
    ("48x0e + 10x1o + 10x1e + 10x0o", "48x0o + 48x0e", 99),
]


@pytest.mark.parametrize("in_irreps,out_irreps,n_feat", SIGS)
@pytest.mark.parametrize("groups", ["per_term", "one"])
def test_schedule_matches_host_tp(in_irreps, out_irreps, n_feat, groups):
    torch.manual_seed(0)
    layer = TensorProductConvLayer(in_irreps, "1x0e + 1x1o + 1x2e", out_irreps, n_feat,
                                   hidden_features=n_feat).double().eval()
    for p in layer.fc.parameters():
        torch.nn.init.normal_(p, std=0.3)
    tp, fc = layer.tp, layer.fc
    n_terms = len(tp.out_irreps.terms)
    plan = EdgeTilePlan(tp, fc, None if groups == "per_term" else [list(range(n_terms))])
    E = 70
    ea = torch.randn(E, n_feat, dtype=torch.float64)
    x1 = torch.randn(E, tp.in_irreps.dim, dtype=torch.float64)
    x2 = torch.randn(E, tp.sh_irreps.dim, dtype=torch.float64)
    with torch.no_grad():
        ref = tp(x1, x2, fc(ea))
    got = emulate(plan, ea, x1, x2)
    assert got.shape == ref.shape
    assert (got - ref).abs().max() <= 1e-12 * ref.abs().max()
