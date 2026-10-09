"""Exact pair search (tt_bio.tfg.neighbors) and the accelerated VinaSteric path, against brute force and the dense path."""
import torch

from tt_bio.tfg.neighbors import pairs_within
from tt_bio.tfg.potentials import VinaStericPotential


def _cloud(n, box, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, n, 3, generator=g) * box


def test_pairs_within_is_brute_force():
    Q, P = _cloud(300, 25.0, 0), _cloud(400, 25.0, 1)
    s, i, j = pairs_within(Q, P, 3.1)
    got = set(zip(s.tolist(), i.tolist(), j.tolist()))
    d = torch.cdist(Q, P)
    want = set(map(tuple, torch.nonzero(d < 3.1).tolist()))
    assert got == want and len(want) > 100


def _complex(seed=0, n=600):
    """Three chains packed into one box so inter-chain clashes exist; element C/N/O/S."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(3, n, 3, generator=g) * 22.0
    elem = torch.zeros(n, 128)
    elem[torch.arange(n), torch.tensor([6, 7, 8, 16])[torch.randint(0, 4, (n,), generator=g)]] = 1.0
    tok = torch.arange(n)
    asym = torch.cat([torch.zeros(250), torch.ones(200), torch.full((150,), 2.0)]).long()
    return X, {"asym_id": asym, "atom_to_token_idx": tok, "ref_element": elem}


def test_vina_sparse_equals_dense_bitwise():
    X, f = _complex()
    pot = VinaStericPotential()
    for step in (0.0, 0.05, 3.0):        # build, reuse inside the skin, rebuild
        Y = X + step * torch.randn(X.shape, generator=torch.Generator().manual_seed(7))
        e0, g0 = pot.energy_and_grad(Y, f, {"buffer": 0.225, "core": "off"})
        e1, g1 = pot.energy_and_grad(Y, f, {"buffer": 0.225, "core": "auto"})
        assert torch.equal(e0, e1) and torch.equal(g0, g1)
        assert float(e0.abs().sum()) > 0
