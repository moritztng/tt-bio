"""Exact pair search (tt_bio.tfg.neighbors) and the accelerated VinaSteric path, against brute force and the dense path."""
import torch

from tt_bio.tfg.neighbors import BoundField, pairs_within
from tt_bio.tfg.potentials import VinaStericPotential


def _cloud(n, box, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, n, 3, generator=g) * box


def test_pairs_within_is_brute_force():
    # the second query cloud reaches well outside P's grid on every side
    for Q, P in ((_cloud(300, 25.0, 0), _cloud(400, 25.0, 1)), (_cloud(500, 45.0, 2) - 10.0, _cloud(400, 25.0, 3))):
        s, i, j = pairs_within(Q, P, 3.1)
        got = list(zip(s.tolist(), i.tolist(), j.tolist()))
        want = set(map(tuple, torch.nonzero(torch.cdist(Q, P) < 3.1).tolist()))
        assert len(got) == len(set(got)) and set(got) == want and len(want) > 100


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


def test_residue_distances_pruned_equals_cdist():
    """The pruned epitope distances equal residue_distances on random rigid poses: values and nearest-pair slots."""
    from tt_bio.tfg.epitope import ResidueDistances, residue_distances

    g = torch.Generator().manual_seed(4)
    body = torch.rand(2, 300, 3, generator=g) * 20.0                 # paratope, 2 samples, 30 residues of 10 atoms
    token = torch.arange(300) // 10
    epi = torch.rand(2, 6 * 5, 3, generator=g) * 20.0 + 8.0         # 6 epitope residues x 5 slots
    slots = torch.arange(30).view(6, 5)
    valid = torch.ones(6, 5, dtype=torch.bool)
    valid[2, 3:] = False
    rd = ResidueDistances(body, token, slots, valid)
    for seed in range(3):
        rot = torch.linalg.matrix_exp(torch.cross(torch.eye(3).expand(2, 3, 3), torch.randn(2, 1, 3, generator=g)
                                                  .expand(2, 3, 3), dim=-1))
        posed = body @ rot.transpose(1, 2) + torch.randn(2, 1, 3, generator=g) * 4.0
        d0, a0, p0, _ = residue_distances(posed, epi, slots, valid, torch.arange(300))
        d1, a1, p1 = rd(posed, epi, torch.arange(2), slots=True)
        assert torch.equal(d0, d1) and torch.equal(a0, a1) and torch.equal(p0, p1)


def test_classify_rows_equals_classify_off_grid_too():
    """classify_rows clamps off-grid points onto the outermost voxel layer, which must be inf (no atom reaches it)."""
    g = torch.Generator().manual_seed(3)
    fixed = torch.rand(3, 400, 3, generator=g) * 30
    rb = torch.rand(400, generator=g) + 1.2
    field = BoundField(fixed, rb, 1.9)
    grid = field.soft.view(3, *(int(d) for d in field.dims))
    for face in (grid[:, 0], grid[:, -1], grid[:, :, 0], grid[:, :, -1], grid[..., 0], grid[..., -1]):
        assert torch.isinf(face).all()
    X = torch.rand(3, 20000, 3, generator=g) * 60 - 15
    ra = torch.rand(20000, generator=g) + 1.2
    samples = torch.tensor([2, 0, 1])
    far, severe = field.classify(X.reshape(-1, 3), samples.repeat_interleave(20000), ra.repeat(3))
    far_r, severe_r = field.classify_rows(X, samples, ra)
    assert torch.equal(far, far_r) and torch.equal(severe, severe_r)
    assert far.any() and (~far).any() and severe.any()
