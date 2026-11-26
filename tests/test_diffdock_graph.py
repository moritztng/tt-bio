"""The static edge sets reproduce upstream's dynamic radius graphs, on real geometry.

The fixture is ``tests/data/diffdock_graph/8G43_ZU6_steps.npz``, dumped from upstream
DiffDock-L on CPU at diffusion steps 0, 8 and 18 of a real 19-step run
(``scripts/diffdock_port/dump_graph_golden.py``, reference environment with torch_cluster and
e3nn). It carries the coordinates the sampler reached, the sigma-dependent cross cutoff, every
edge torch_cluster returned for pose 0, and a sample of upstream's own edge features.

Three steps, not one, because the cross graph is the thing that moves: the cutoff is
``3*tr_sigma + 20``, so step 0 is the complete bipartite graph, step 8 is 96.6 % of it and
step 18 is 69.5 % (ARCH.md 6.1). A mask that is wrong in one direction would still pass at
step 0, where everything is inside the cutoff.

What is graded here:

* the masked static edge set **is** upstream's edge set, for all three graphs, as a set;
* the port's own edge features (harmonics, distance expansion) match upstream's to fp32
  rounding on the edges they share -- otherwise an identical edge set would prove nothing;
* the node update out of a convolution is the same whether it is computed on upstream's
  dynamic edge list or on the padded static set with the mask. That is the claim the traced
  loop rests on, and it fails if the ``mean`` divisor counts padded or masked-out edges.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock import graph as G  # noqa: E402
from tt_bio.diffdock.conv import TensorProductConvLayer, bucket  # noqa: E402
from tt_bio.diffdock.features import GaussianSmearing  # noqa: E402
from tt_bio.diffdock.harmonics import spherical_harmonics  # noqa: E402

FIXTURE = Path(__file__).parent / "data/diffdock_graph/8G43_ZU6_steps.npz"
STEPS = (0, 8, 18)
LIG_MAX_RADIUS = 5.0
CROSS_MAX_DISTANCE = 80.0
SH_LS = (0, 1, 2)
# fp32 rounding on a unit-norm harmonic and on a radial basis in [0, 1]
FEATURE_TOL = 2e-6


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE.exists():
        pytest.skip(f"missing {FIXTURE}; regenerate with scripts/diffdock_port/"
                    "dump_graph_golden.py in the reference environment")
    return _npz(FIXTURE)


def _npz(path):
    import numpy as np
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out


def step_data(golden, step):
    """One step's tensors, as torch, with the port's static graphs already built."""
    p = f"s{step}_"
    t = {k[len(p):]: torch.as_tensor(v) for k, v in golden.items()
         if isinstance(k, str) and k.startswith(p)}
    t["lig_pos"] = t["lig_pos"].float()
    t["rec_pos"] = t["rec_pos"].float()
    n_lig, n_rec = t["lig_pos"].shape[0], t["rec_pos"].shape[0]
    bonds = t["lig_bond_edge_index"].long()
    n_rot = int(t["lig_edge_mask"].sum())
    static = G.StaticGraphs.build(n_lig, n_rec, bonds, n_rot_bonds=n_rot)
    t["static"], t["n_lig"], t["n_rec"], t["n_rot"] = static, n_lig, n_rec, n_rot
    t["cutoff"] = float(t["cross_cutoff"][0])
    return t


def as_set(edge_index):
    return {(int(a), int(b)) for a, b in zip(*edge_index)}


# --------------------------------------------------------------- edge sets

@pytest.mark.parametrize("step", STEPS)
def test_cross_mask_selects_upstreams_cross_edges(golden, step):
    d = step_data(golden, step)
    s = d["static"]
    mask = s.cross_mask(d["lig_pos"], d["rec_pos"], d["cutoff"])
    kept = s.cross_edge_index[:, mask.bool()]
    assert as_set(kept) == as_set(d["cross_edge_index"])
    # and the padding never survives the mask
    assert mask[s.n_lig * s.n_rec:].sum() == 0


@pytest.mark.parametrize("step", STEPS)
def test_cross_completeness_follows_the_schedule(golden, step):
    """Sanity on the fixture itself: the cross graph anneals, it is not always complete."""
    d = step_data(golden, step)
    frac = d["cross_edge_index"].shape[1] / (d["n_lig"] * d["n_rec"])
    assert 0.5 <= frac <= 1.0
    if step == 0:
        assert frac == 1.0


@pytest.mark.parametrize("step", STEPS)
def test_ligand_mask_selects_upstreams_radius_edges(golden, step):
    d = step_data(golden, step)
    s = d["static"]
    n_bond = s.lig_n_bond_edges
    mask = s.lig_mask(d["lig_pos"], LIG_MAX_RADIUS)
    assert mask[:n_bond].min() == 1, "a bond edge is never masked out"
    kept = s.lig_edge_index[:, mask.bool()]
    assert as_set(kept[:, n_bond:]) == as_set(d["lig_edge_index"][:, n_bond:])
    assert as_set(kept[:, :n_bond]) == as_set(d["lig_edge_index"][:, :n_bond])
    assert kept.shape[1] == d["lig_edge_index"].shape[1], "bonded pairs are counted twice"


@pytest.mark.parametrize("step", STEPS)
def test_torsion_mask_selects_upstreams_bond_conv_edges(golden, step):
    d = step_data(golden, step)
    s = d["static"]
    bonds = d["tor_bonds"].long()
    bond_pos = (d["lig_pos"][bonds[0]] + d["lig_pos"][bonds[1]]) / 2
    mask = s.tor_mask(d["lig_pos"], bond_pos, LIG_MAX_RADIUS)
    kept = s.tor_edge_index[:, mask.bool()]
    assert as_set(kept) == as_set(d["tor_edge_index"])


def test_padding_is_a_multiple_of_32_and_points_at_a_free_node(golden):
    d = step_data(golden, STEPS[0])
    s = d["static"]
    for name, ei, pad in (("lig", s.lig_edge_index, s.lig_pad_node),
                          ("cross", s.cross_edge_index, s.lig_pad_node),
                          ("tor", s.tor_edge_index, s.n_bond_padded - 1)):
        assert ei.shape[1] % 32 == 0, name
        assert ei[0, -1] == pad, name
    assert s.n_lig_padded % 32 == 0 and s.n_lig_padded > s.n_lig
    assert s.n_rec_padded % 32 == 0 and s.n_rec_padded > s.n_rec


# --------------------------------------------------------------- edge features

@pytest.mark.parametrize("step", STEPS)
def test_harmonics_match_e3nn_on_upstreams_own_edges(golden, step):
    d = step_data(golden, step)
    rows = d["cross_sample_rows"].long()
    ei = d["cross_edge_index"][:, rows].long()
    vec = d["rec_pos"][ei[1]] - d["lig_pos"][ei[0]]
    assert torch.allclose(spherical_harmonics(vec, SH_LS), d["cross_edge_sh_sample"],
                          atol=FEATURE_TOL, rtol=0)
    assert torch.allclose(spherical_harmonics(-vec, SH_LS), d["rev_cross_edge_sh_sample"],
                          atol=FEATURE_TOL, rtol=0)


@pytest.mark.parametrize("step", STEPS)
def test_cross_distance_expansion_matches_upstream(golden, step):
    d = step_data(golden, step)
    rows = d["cross_sample_rows"].long()
    ei = d["cross_edge_index"][:, rows].long()
    vec = d["rec_pos"][ei[1]] - d["lig_pos"][ei[0]]
    expansion = GaussianSmearing(0.0, CROSS_MAX_DISTANCE, 64)
    attr = torch.cat([d["lig_node_sigma_emb"][ei[0]], expansion(vec.norm(dim=-1))], dim=1)
    assert torch.allclose(attr, d["cross_edge_attr_sample"], atol=FEATURE_TOL, rtol=0)


def test_harmonics_are_finite_for_a_zero_length_edge():
    """A complete edge set contains pairs the mask will drop, including zero-length ones.

    e3nn normalises with an epsilon for the same reason. A NaN here would survive the mask:
    0 * NaN is NaN, and the node sum would be NaN for every node the pad edge touches.
    """
    sh = spherical_harmonics(torch.zeros(3, 3), SH_LS)
    assert torch.isfinite(sh).all()
    assert sh[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert sh[:, 1:].abs().max() == 0


# --------------------------------------------------------------- the node update

def _conv(in_dim, n_edge_features, seed=0):
    torch.manual_seed(seed)
    layer = TensorProductConvLayer(
        in_irreps=f"{in_dim}x0e", sh_irreps="1x0e+1x1o+1x2e",
        out_irreps=f"{in_dim}x0e+4x1o", n_edge_features=n_edge_features, edge_groups=2,
        hidden_features=32, batch_norm=False, residual=False)
    layer.eval()
    for p in layer.parameters():          # BatchNorm buffers aside, the FC block starts tiny
        torch.nn.init.normal_(p, std=0.3)
    return layer


@pytest.mark.parametrize("step", STEPS)
def test_node_update_matches_the_dynamic_graph(golden, step):
    """The deliverable: same node update, dynamic edge list vs static padded set + mask.

    Two edge groups, ligand-ligand and ligand<-receptor, which are the two the port makes
    static; the receptor graph is a k-NN graph built once and never rebuilt, and the reverse
    cross group is the cross group with its rows swapped. Weights are random rather than the
    checkpoint's on purpose: the claim is about the graph, and random weights make every
    component of every edge message matter instead of whatever the trained filter suppresses.
    """
    d = step_data(golden, step)
    s, n_lig, n_rec = d["static"], d["n_lig"], d["n_rec"]
    in_dim, n_edge_features = 16, 24
    layer = _conv(in_dim, n_edge_features)

    torch.manual_seed(step + 1)
    node_attr_real = torch.randn(n_lig + n_rec, in_dim)
    lig_expansion = GaussianSmearing(0.0, LIG_MAX_RADIUS, n_edge_features)
    cross_expansion = GaussianSmearing(0.0, CROSS_MAX_DISTANCE, n_edge_features)

    def run(lig_ei, cross_ei, node_attr, n_nodes, rec_offset, lig_pos, rec_pos,
            lig_mask=None, cross_mask=None):
        lig_vec = lig_pos[lig_ei[1]] - lig_pos[lig_ei[0]]
        cross_vec = rec_pos[cross_ei[1] - rec_offset] - lig_pos[cross_ei[0]]
        edge_index = torch.cat([lig_ei, cross_ei], dim=1)
        edge_attr = [lig_expansion(lig_vec.norm(dim=-1)),
                     cross_expansion(cross_vec.norm(dim=-1))]
        edge_sh = torch.cat([spherical_harmonics(lig_vec, SH_LS),
                             spherical_harmonics(cross_vec, SH_LS)])
        mask = None
        if lig_mask is not None:
            mask = torch.cat([lig_mask, cross_mask])
        return layer(node_attr, edge_index, edge_attr, edge_sh, out_nodes=n_nodes,
                     edge_mask=mask)

    # upstream's dynamic edge lists, on the unpadded node axis
    lig_dyn = d["lig_edge_index"].long()
    cross_dyn = d["cross_edge_index"].long().clone()
    cross_dyn[1] += n_lig
    dynamic = run(lig_dyn, cross_dyn, node_attr_real, n_lig + n_rec, n_lig,
                  d["lig_pos"], d["rec_pos"])

    # the port: static padded sets, a mask from the same coordinates, padded node axis
    node_attr = torch.zeros(s.n_lig_padded + s.n_rec_padded, in_dim)
    node_attr[:n_lig] = node_attr_real[:n_lig]
    node_attr[s.rec_offset:s.rec_offset + n_rec] = node_attr_real[n_lig:]
    lig_pos_padded = torch.zeros(s.n_lig_padded, 3)
    lig_pos_padded[:n_lig] = d["lig_pos"]
    rec_pos_padded = torch.zeros(s.n_rec_padded, 3)
    rec_pos_padded[:n_rec] = d["rec_pos"]
    static_out = run(s.lig_edge_index, s.cross_edge_index_global(), node_attr,
                     s.n_lig_padded + s.n_rec_padded, s.rec_offset,
                     lig_pos_padded, rec_pos_padded,
                     lig_mask=s.lig_mask(lig_pos_padded, LIG_MAX_RADIUS),
                     cross_mask=s.cross_mask(lig_pos_padded, rec_pos_padded, d["cutoff"]))

    real = torch.cat([static_out[:n_lig], static_out[s.rec_offset:s.rec_offset + n_rec]])
    assert torch.isfinite(real).all()
    scale = dynamic.abs().max()
    err = (real - dynamic).abs().max() / scale
    assert err < 1e-6, f"step {step}: max relative deviation {err:.3e}"


# --------------------------------------------------- the same claim, without the fixture

def test_node_update_matches_a_brute_force_dynamic_graph():
    """The graph claim on synthetic geometry, so it is checked even without the dump.

    The fixture above is what pins the port against torch_cluster and e3nn; this runs
    anywhere and catches the ways the static path can break on its own -- a pad edge landing
    on a real node, a divisor counting masked edges, a NaN from a zero-length pad edge.
    """
    torch.manual_seed(0)
    n_lig, n_rec, n_rot = 21, 96, 5
    cutoff, in_dim, n_edge_features = 25.0, 16, 24
    lig_pos, rec_pos = torch.randn(n_lig, 3) * 2.0, torch.randn(n_rec, 3) * 8.0
    chain = torch.stack([torch.arange(n_lig - 1), torch.arange(1, n_lig)])
    bonds = torch.cat([chain, chain.flip(0)], dim=1)
    s = G.StaticGraphs.build(n_lig, n_rec, bonds, n_rot_bonds=n_rot)
    assert s.lig_edge_index.shape[1] > bonds.shape[1] + n_lig * (n_lig - 1), "no pad edges"

    layer = _conv(in_dim, n_edge_features)
    node_attr_real = torch.randn(n_lig + n_rec, in_dim)
    lig_expansion = GaussianSmearing(0.0, LIG_MAX_RADIUS, n_edge_features)
    cross_expansion = GaussianSmearing(0.0, CROSS_MAX_DISTANCE, n_edge_features)

    def run(lig_ei, cross_ei, node_attr, n_nodes, offset, lp, rp, mask=None):
        lig_vec = lp[lig_ei[1]] - lp[lig_ei[0]]
        cross_vec = rp[cross_ei[1] - offset] - lp[cross_ei[0]]
        return layer(node_attr, torch.cat([lig_ei, cross_ei], dim=1),
                     [lig_expansion(lig_vec.norm(dim=-1)),
                      cross_expansion(cross_vec.norm(dim=-1))],
                     torch.cat([spherical_harmonics(lig_vec, SH_LS),
                                spherical_harmonics(cross_vec, SH_LS)]),
                     out_nodes=n_nodes, edge_mask=mask)

    inside = (rec_pos[None] / cutoff - lig_pos[:, None] / cutoff).norm(dim=-1) <= 1
    cross_dyn = inside.nonzero().T
    cross_dyn[1] += n_lig
    near = (lig_pos[None] - lig_pos[:, None]).norm(dim=-1) <= LIG_MAX_RADIUS
    near.fill_diagonal_(False)
    dynamic = run(torch.cat([bonds, near.nonzero().T], dim=1), cross_dyn, node_attr_real,
                  n_lig + n_rec, n_lig, lig_pos, rec_pos)

    node_attr = torch.zeros(s.n_lig_padded + s.n_rec_padded, in_dim)
    node_attr[:n_lig] = node_attr_real[:n_lig]
    node_attr[s.rec_offset:s.rec_offset + n_rec] = node_attr_real[n_lig:]
    lp = torch.zeros(s.n_lig_padded, 3)
    lp[:n_lig] = lig_pos
    rp = torch.zeros(s.n_rec_padded, 3)
    rp[:n_rec] = rec_pos
    out = run(s.lig_edge_index, s.cross_edge_index_global(), node_attr,
              s.n_lig_padded + s.n_rec_padded, s.rec_offset, lp, rp,
              mask=torch.cat([s.lig_mask(lp, LIG_MAX_RADIUS),
                              s.cross_mask(lp, rp, cutoff)]))
    real = torch.cat([out[:n_lig], out[s.rec_offset:s.rec_offset + n_rec]])
    assert torch.isfinite(real).all()
    assert (real - dynamic).abs().max() / dynamic.abs().max() < 1e-6


def test_a_masked_edge_is_not_counted_in_the_mean():
    """The divisor is the mask's sum, not the edge count. Pin it directly.

    With ``reduce='mean'`` and a mask of zeros on half the edges, counting all the edges
    would halve every node's output. This is the single mistake that makes a static graph
    look right -- same edges, same messages -- and give numbers that are off by a factor
    that differs per node.
    """
    torch.manual_seed(0)
    n, in_dim, n_edge_features = 8, 16, 24
    layer = _conv(in_dim, n_edge_features)
    node_attr = torch.randn(n, in_dim)
    ei = G.complete_pairs(n, n, exclude_diagonal=True)
    keep = torch.zeros(ei.shape[1], dtype=torch.bool)
    keep[::2] = True
    sh = spherical_harmonics(torch.randn(ei.shape[1], 3), SH_LS)
    attr = torch.randn(ei.shape[1], n_edge_features)
    half = layer(node_attr, ei[:, keep], [attr[keep], attr[:0]], sh[keep])
    masked = layer(node_attr, ei, [attr, attr[:0]], sh, edge_mask=keep.float())
    assert torch.allclose(half, masked, atol=1e-6, rtol=0)


# --------------------------------------------------------------- pose batching

def _pose_system(n_lig, n_rec, n_poses, seed=0):
    """One complex, *n_poses* poses: same topology and receptor, different coordinates."""
    torch.manual_seed(seed)
    lig = torch.randn(n_poses, n_lig, 3) * 2.0 + torch.randn(n_poses, 1, 3) * 3.0
    rec = (torch.randn(n_rec, 3) * 8.0).repeat(n_poses, 1, 1)
    chain = torch.stack([torch.arange(n_lig - 1), torch.arange(1, n_lig)])
    bonds = torch.cat([chain, chain.flip(0)], dim=1)
    return lig, rec, bonds


def test_pose_batch_is_block_diagonal():
    """Every pose's edges stay inside its own node block -- the one thing batching can break.

    tt-atom's note applies unchanged (``tt_atom/disjoint.py``): build each system's edge set
    separately and offset it, never a radius graph over the concatenated coordinates, or an
    edge wires one pose's ligand to another pose's receptor. Here the edge set is complete
    per pose, so the check is that the offsets partition it.
    """
    n_lig, n_rec, n_poses = 7, 11, 4
    _, _, bonds = _pose_system(n_lig, n_rec, n_poses)
    s = G.StaticGraphs.build(n_lig, n_rec, bonds, n_rot_bonds=3, n_poses=n_poses)
    real_cross = s.cross_edge_index[:, :n_lig * n_rec * n_poses]
    assert (real_cross[0] // n_lig == real_cross[1] // n_rec).all()
    real_lig = s.lig_edge_index[:, :s.lig_n_bond_edges + n_lig * (n_lig - 1) * n_poses]
    assert (real_lig[0] // n_lig == real_lig[1] // n_lig).all()
    real_tor = s.tor_edge_index[:, :3 * n_lig * n_poses]
    assert (real_tor[0] // 3 == real_tor[1] // n_lig).all()
    # every pose contributes the same edges, shifted
    one = G.StaticGraphs.build(n_lig, n_rec, bonds, n_rot_bonds=3, n_poses=1)
    per_pose = one.cross_edge_index[:, :n_lig * n_rec]
    for p in range(n_poses):
        block = real_cross[:, p * n_lig * n_rec:(p + 1) * n_lig * n_rec]
        assert torch.equal(block - torch.tensor([[p * n_lig], [p * n_rec]]), per_pose)


def test_batched_node_update_equals_the_poses_run_one_at_a_time():
    """A pose batch is ten independent complexes in one call, and must stay that way.

    Same convolution, same coordinates: the batched output's slice for pose p has to be what
    running pose p alone gives. It would not be if the mean divisor leaked across poses, if a
    pad edge pointed into another pose's block, or if a per-pose cutoff were applied to the
    wrong edges -- so the cutoffs here differ per pose, which is the shape upstream's
    ``cross_cutoff`` tensor has.
    """
    n_lig, n_rec, n_poses = 9, 13, 4
    in_dim, n_edge_features = 16, 24
    lig, rec, bonds = _pose_system(n_lig, n_rec, n_poses, seed=3)
    cutoffs = torch.tensor([30.0, 24.0, 21.0, 20.5])
    layer = _conv(in_dim, n_edge_features)
    lig_expansion = GaussianSmearing(0.0, LIG_MAX_RADIUS, n_edge_features)
    cross_expansion = GaussianSmearing(0.0, CROSS_MAX_DISTANCE, n_edge_features)
    torch.manual_seed(11)
    lig_attr = torch.randn(n_poses, n_lig, in_dim)
    rec_attr = torch.randn(n_rec, in_dim).repeat(n_poses, 1, 1)

    def run(s, lig_pos, rec_pos, lig_attr, rec_attr, cutoff):
        node_attr = torch.zeros(s.n_lig_padded + s.n_rec_padded, in_dim)
        node_attr[:s.n_lig_total] = lig_attr.reshape(-1, in_dim)
        node_attr[s.rec_offset:s.rec_offset + s.n_rec_total] = rec_attr.reshape(-1, in_dim)
        lp = torch.zeros(s.n_lig_padded, 3)
        lp[:s.n_lig_total] = lig_pos.reshape(-1, 3)
        rp = torch.zeros(s.n_rec_padded, 3)
        rp[:s.n_rec_total] = rec_pos.reshape(-1, 3)
        cross_ei = s.cross_edge_index_global()
        lig_vec = lp[s.lig_edge_index[1]] - lp[s.lig_edge_index[0]]
        cross_vec = rp[cross_ei[1] - s.rec_offset] - lp[cross_ei[0]]
        out = layer(node_attr, torch.cat([s.lig_edge_index, cross_ei], dim=1),
                    [lig_expansion(lig_vec.norm(dim=-1)),
                     cross_expansion(cross_vec.norm(dim=-1))],
                    torch.cat([spherical_harmonics(lig_vec, SH_LS),
                               spherical_harmonics(cross_vec, SH_LS)]),
                    out_nodes=s.n_lig_padded + s.n_rec_padded,
                    edge_mask=torch.cat([s.lig_mask(lp, LIG_MAX_RADIUS),
                                         s.cross_mask(lp, rp, cutoff)]))
        return (out[:s.n_lig_total].reshape(s.n_poses, s.n_lig, -1),
                out[s.rec_offset:s.rec_offset + s.n_rec_total].reshape(s.n_poses, s.n_rec, -1))

    batched = run(G.StaticGraphs.build(n_lig, n_rec, bonds, n_poses=n_poses),
                  lig, rec, lig_attr, rec_attr, cutoffs)
    single = G.StaticGraphs.build(n_lig, n_rec, bonds, n_poses=1)
    scale = batched[0].abs().max()
    assert scale > 0
    for p in range(n_poses):
        one = run(single, lig[p], rec[p], lig_attr[p], rec_attr[p], cutoffs[p])
        for batch_out, one_out in zip(batched, one):
            # the receptor rows are identically zero here on both sides: both edge groups
            # scatter to ligand nodes, and the group that feeds receptor nodes is the
            # reverse-cross one, which is the cross group with its rows swapped.
            err = (batch_out[p] - one_out[0]).abs().max() / scale
            assert err < 1e-6, f"pose {p}: {err:.3e}"
