"""ESMFold2's relative-position encoding by table lookup (`relpos_lookup`) against the one-hot features
times the weight, in float64, on a complex with two chains, a symmetric copy and a multi-token residue."""
import pytest

torch = pytest.importorskip("torch")

from tt_bio.esmfold2 import relpos_features, relpos_lookup  # noqa: E402
import torch.nn.functional as F  # noqa: E402


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


def test_rows_are_the_lookup():
    """`relpos_rows` (codes + one row per distinct code, what the device embeds) is the lookup bit for bit."""
    from tt_bio.esmfold2 import relpos_rows

    g = torch.Generator().manual_seed(1)
    ri = torch.cat([torch.arange(30), torch.tensor([4, 4, 4]), torch.arange(20)])
    ti = torch.arange(ri.numel())
    asym = torch.cat([torch.zeros(33), torch.ones(20)]).long()
    sym = torch.zeros(53).long()
    ent = torch.cat([torch.zeros(33), torch.ones(20)]).long()
    args = [t.unsqueeze(0) for t in (ri, asym, sym, ent, ti)]
    w = torch.randn(32, 2 * 66 + 1 + 6, generator=g)
    p = relpos_rows(w, *args)
    assert p.shape == (1, 53, 53, 32) and p.rows.shape[0] < 53 * 53
    assert p.float() is p and p.detach() is p
    # the per-pair sum relpos_lookup did before the rows: same fp32 adds in the same order
    feats = relpos_features(*args)
    n, t = 66, w.t()
    ref = F.embedding(feats[..., :n].argmax(-1), t[:n])
    ref += F.embedding(feats[..., n:2 * n].argmax(-1), t[n:2 * n])
    ref += F.embedding(feats[..., 2 * n + 1:].argmax(-1), t[2 * n + 1:])
    ref += feats[..., 2 * n].bool().unsqueeze(-1) * t[2 * n]
    assert torch.equal(p.host(), ref)
