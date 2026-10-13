"""ESMFold2's relative-position encoding by table lookup (`relpos_lookup`) against the one-hot features
times the weight, in float64, on a complex with two chains, a symmetric copy and a multi-token residue."""
import pytest

torch = pytest.importorskip("torch")

from tt_bio.esmfold2 import relpos_features, relpos_lookup  # noqa: E402


def test_lookup_matches_one_hot_product():
    g = torch.Generator().manual_seed(0)
    # chain A 0..39 with residue 7 split over 3 tokens, chain B (copy of A's entity, sym 1), chain C (entity 1)
    ri = torch.cat([torch.arange(40), torch.tensor([7, 7]), torch.arange(30), torch.arange(5)])
    ti = torch.arange(ri.numel())
    asym = torch.cat([torch.zeros(42), torch.ones(30), torch.full((5,), 2)]).long()
    sym = torch.cat([torch.zeros(42), torch.ones(30), torch.zeros(5)]).long()
    ent = torch.cat([torch.zeros(72), torch.ones(5)]).long()
    args = [t.unsqueeze(0) for t in (ri, asym, sym, ent, ti)]
    w = torch.randn(64, 2 * 66 + 1 + 6, generator=g)
    ref = relpos_features(*args).double() @ w.t().double()
    out = relpos_lookup(w, *args)
    assert out.shape == ref.shape and out.dtype == torch.float32
    assert (out.double() - ref).abs().max().item() < 1e-5
