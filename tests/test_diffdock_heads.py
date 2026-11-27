"""The tr/rot score heads and the edge spherical harmonics.

Real-weight grade (``scripts/diffdock_port/heads_parity.py``, reference env, released
checkpoint, upstream's own conv/smearing/so3 modules): tr max relative error 1.23e-07, rot
1.65e-07; spherical harmonics bit-identical to e3nn for both ``0e+1o+2e`` and ``2e``. The bar
is fp32 rounding through one conv layer and two small MLPs, i.e. ~1e-7; above 1e-5 is a real
difference.

Here, without e3nn: rotation equivariance of the whole head in float64 -- the property the
docking actually depends on, and the one the norm-and-rescale step could break -- plus the
``fixed_center_conv`` trap and the harmonics' own invariants.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock.harmonics import spherical_harmonics
from tt_bio.diffdock.heads import CenterHead, TorsionHead
from tt_bio.diffdock.irreps import Irreps

IN = "8x0e + 3x1o + 3x1e + 3x0o"
SH = "1x0e + 1x1o + 1x2e"
NS, SIG, DIST = 8, 6, 10


def _rot(a, b, c):
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    rz = lambda c_, s_: torch.tensor([[c_, -s_, 0.], [s_, c_, 0.], [0., 0., 1.]], dtype=torch.float64)
    ry = torch.tensor([[cb, 0., sb], [0., 1., 0.], [-sb, 0., cb]], dtype=torch.float64)
    return rz(ca, sa) @ ry @ rz(cc, sc)


def _rotate(irreps, x, r):
    out = x.clone()
    for term, sl in zip(Irreps.parse(irreps), Irreps.parse(irreps).slices()):
        if term.l == 1:
            out[:, sl] = (x[:, sl].reshape(-1, term.mul, 3) @ r.T).reshape(-1, term.mul * 3)
    return out


def _head(**kw):
    torch.manual_seed(2)
    h = CenterHead(in_irreps=IN, sh_irreps=SH, ns=NS, distance_embed_dim=DIST,
                   sigma_embed_dim=SIG, **kw).double().eval()
    bn = h.final_conv.batch_norm
    with torch.no_grad():
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.normal_(1.0, 0.2)
    return h


def _inputs(sizes=(5, 12, 9)):
    g = torch.Generator().manual_seed(4)
    batch = torch.cat([torch.full((s,), i) for i, s in enumerate(sizes)])
    n = batch.numel()
    pos = torch.randn(n, 3, generator=g, dtype=torch.float64) * 3
    feats = torch.randn(n, Irreps.parse(IN).dim, generator=g, dtype=torch.float64)
    emb = torch.randn(len(sizes), SIG, generator=g, dtype=torch.float64)
    sig = torch.tensor([0.5, 2.0, 7.0], dtype=torch.float64)
    rsig = torch.tensor([0.1, 0.4, 1.2], dtype=torch.float64)
    return batch, pos, feats, emb, sig, rsig


def test_spherical_harmonics_basic_invariants():
    v = torch.randn(200, 3, dtype=torch.float64)
    sh = spherical_harmonics(v)
    assert sh.shape == (200, 9)
    # component normalisation: |Y_l|^2 = 2l+1 on the unit sphere
    assert torch.allclose(sh[:, 1:4].pow(2).sum(1), torch.full((200,), 3.0, dtype=torch.float64))
    assert torch.allclose(sh[:, 4:9].pow(2).sum(1), torch.full((200,), 5.0, dtype=torch.float64))
    # scale-free (normalize=True), and l=2 alone is the tail of the full set
    assert torch.allclose(spherical_harmonics(7.5 * v), sh)
    assert torch.equal(spherical_harmonics(v, (2,)), sh[:, 4:9])


def test_a_zero_edge_gives_zeros_not_nan():
    sh = spherical_harmonics(torch.zeros(1, 3))
    assert torch.isfinite(sh).all()
    assert sh[0, 0] == 1 and torch.equal(sh[0, 1:], torch.zeros(8))


def test_harmonics_beyond_l2_are_refused():
    with pytest.raises(ValueError, match="l=2"):
        spherical_harmonics(torch.ones(1, 3), (3,))


def test_the_head_is_rotation_equivariant_in_float64():
    """Rotate the ligand and its l=1 features: both scores rotate with it.

    Rotation only: the head adds a 1o and a 1e vector (upstream's ``odd_parity: false``),
    which is not parity-equivariant by design, so a mirror test would be testing upstream's
    modelling choice rather than this port.
    """
    h = _head()
    batch, pos, feats, emb, sig, rsig = _inputs()
    r = _rot(0.7, -1.3, 2.1)
    with torch.no_grad():
        tr, rot = h(feats, pos, batch, 3, emb[batch], emb, sig, rsig)
        tr2, rot2 = h(_rotate(IN, feats, r), pos @ r.T + torch.tensor([3.0, -1.0, 2.0],
                      dtype=torch.float64), batch, 3, emb[batch], emb, sig, rsig)
    for a, b in ((tr2, tr @ r.T), (rot2, rot @ r.T)):
        rel = (a - b).abs().max() / b.abs().max()
        assert rel < 1e-12, f"equivariance broken at {rel:.3e}"


def test_fixed_center_conv_reads_the_atom_not_the_graph_id():
    """The released model sets ``fixed_center_conv``. Upstream's constructor default does not,
    and that branch indexes node features by *graph* id. Same weights, different answer --
    so the default here is the released value, and this pins that the two really differ."""
    batch, pos, feats, emb, sig, rsig = _inputs()
    with torch.no_grad():
        a = _head()(feats, pos, batch, 3, emb[batch], emb, sig, rsig)[0]
        b = _head(fixed_center_conv=False)(feats, pos, batch, 3, emb[batch], emb, sig, rsig)[0]
    assert not torch.allclose(a, b, atol=1e-6)
    assert CenterHead.__init__.__kwdefaults__["fixed_center_conv"] is True


def test_scale_by_sigma_divides_tr_and_multiplies_rot():
    batch, pos, feats, emb, sig, rsig = _inputs()
    with torch.no_grad():
        on = _head()(feats, pos, batch, 3, emb[batch], emb, sig, rsig)
        off = _head(scale_by_sigma=False)(feats, pos, batch, 3, emb[batch], emb, sig, rsig)
    assert torch.allclose(on[0] * sig.unsqueeze(1), off[0])
    from tt_bio.diffdock.schedules import so3_score_norm
    assert torch.allclose(on[1], off[1] * so3_score_norm(rsig).double().unsqueeze(1))


# --- the torsion head ------------------------------------------------------------------
#
# Real-weight grade (same script, released checkpoint, same graph both sides): the head
# itself is at 4.71e-07 max relative error, fp32 rounding through a 45-wide tensor product
# and two MLPs. With ``scale_by_sigma`` on it is 9.8e-03 -- and all of that is the sigma
# scaling, whose table is upstream's unseeded Monte Carlo over the wrapped normal: its own
# run-to-run spread is 0.8 % median / 3.6 % max (three processes, measured), so 1 % is inside
# the reference's own noise and is the bar here, not a port defect. The head's bar is 1e-6.

TOR_IN = "8x0e + 3x1o + 3x1e + 3x0o"


def _tor_head(**kw):
    torch.manual_seed(3)
    h = TorsionHead(in_irreps=TOR_IN, sh_irreps=SH, ns=NS, distance_embed_dim=DIST,
                    lig_max_radius=5.0, **kw).double().eval()
    bn = h.tor_bond_conv.batch_norm
    with torch.no_grad():
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.normal_(1.0, 0.2)
    return h


def _tor_inputs(n=24, n_bonds=5):
    g = torch.Generator().manual_seed(6)
    pos = torch.randn(n, 3, generator=g, dtype=torch.float64) * 2
    feats = torch.randn(n, Irreps.parse(TOR_IN).dim, generator=g, dtype=torch.float64)
    bonds = torch.stack([torch.arange(n_bonds) * 2, torch.arange(n_bonds) * 2 + 1])
    bond_pos = (pos[bonds[0]] + pos[bonds[1]]) / 2
    d = (pos.unsqueeze(0) - bond_pos.unsqueeze(1)).norm(dim=-1)
    rows, cols = (d < 5.0).nonzero(as_tuple=True)
    return pos, feats, bonds, torch.stack([rows, cols])


def test_the_torsion_score_is_rotation_invariant():
    """One scalar per bond: rotating the complex must not move it at all. This is the
    property the 45-wide tensor product is there to give, and the one a wrong output layout
    in :class:`FullTensorProduct` would break while every shape still matched."""
    h = _tor_head(scale_by_sigma=False)
    pos, feats, bonds, edge_index = _tor_inputs()
    r = _rot(0.4, 1.1, -0.9)
    with torch.no_grad():
        a = h(feats, pos, bonds, edge_index)
        b = h(_rotate(TOR_IN, feats, r), pos @ r.T + torch.tensor([2., 5., -3.],
              dtype=torch.float64), bonds, edge_index)
    rel = (a - b).abs().max() / a.abs().max()
    assert rel < 1e-12, f"torsion score moved by {rel:.3e} under a rotation"


def test_swapping_a_bonds_two_atoms_does_not_change_its_score():
    """The bond enters only as its midpoint, the *sum* of its endpoints' features, and the
    ``2e`` harmonics of its axis -- and l=2 harmonics are even, so the axis has no direction.
    Reading the axis at l=1 instead would load cleanly and flip sign here."""
    h = _tor_head(scale_by_sigma=False)
    pos, feats, bonds, edge_index = _tor_inputs()
    with torch.no_grad():
        a = h(feats, pos, bonds, edge_index)
        b = h(feats, pos, bonds.flip(0), edge_index)
    assert torch.allclose(a, b, atol=1e-12)


def test_the_final_torsion_layer_has_no_biases():
    """Both linears are ``bias=False`` upstream, and the activation between them is Tanh,
    not the ReLU every other MLP in the model uses."""
    layers = [m for m in _tor_head().tor_final_layer if isinstance(m, torch.nn.Linear)]
    assert len(layers) == 2 and all(m.bias is None for m in layers)
    assert any(isinstance(m, torch.nn.Tanh) for m in _tor_head().tor_final_layer)


def test_scale_by_sigma_is_the_square_root_of_the_torus_table():
    from tt_bio.diffdock.schedules import torus_score_norm
    pos, feats, bonds, edge_index = _tor_inputs()
    sigma = torch.tensor([0.05, 0.3, 0.9, 1.7, 3.0], dtype=torch.float64)
    with torch.no_grad():
        off = _tor_head(scale_by_sigma=False)(feats, pos, bonds, edge_index)
        on = _tor_head()(feats, pos, bonds, edge_index, sigma)
    want = off * torch.as_tensor(torus_score_norm(sigma)).sqrt()
    assert torch.allclose(on, want, rtol=1e-12)


def test_a_bond_with_no_neighbour_inside_the_radius_is_zero_not_nan():
    """The mean divisor is a neighbour count, so an isolated bond divides by zero unless the
    scatter clamps it. Real complexes do not produce one; a padded bond does."""
    h = _tor_head(scale_by_sigma=False)
    pos, feats, bonds, edge_index = _tor_inputs()
    keep = edge_index[0] != 0                       # strip every edge of bond 0
    with torch.no_grad():
        out = h(feats, pos, bonds, edge_index[:, keep])
    assert torch.isfinite(out).all()
    assert out[0].abs().item() < 1e-9
