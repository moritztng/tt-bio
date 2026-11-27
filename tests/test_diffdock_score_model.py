"""The assembled score network.

End-to-end grade with the released v1.1 checkpoint lives in
``scripts/diffdock_port/score_model_grade.py``: it replays dfd-ref's captured upstream
``(tr, rot, tor)`` for 10 poses at several steps of a real rollout, which needs the weights,
the ESM-2 encoder and a PoseBusters structure and so cannot run here.

What this file pins is the wiring that a wrong answer would hide behind:

* the **strict** checkpoint fit -- key names, head regrouping, and the fact that the released
  checkpoint sets ``differentiate_convolutions`` (4 FC weight blocks per conv layer, 2 on the
  last). A single-group assembly loads *nothing* in those layers.
* the receptor cache: reusing it must give the identical answer to recomputing the tower.
* equivariance of the assembled network, which is the campaign's accuracy gate.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock.config import DiffDockConfig, SCORE_PIN  # noqa: E402
from tt_bio.diffdock.score_model import (HEAD_PREFIXES, ScoreModel,  # noqa: E402
                                         radius_edges, remap_checkpoint)


def _config(**kw) -> DiffDockConfig:
    """A tiny score config with the released one's structure and a fraction of its width."""
    base = dict(
        name="test", all_atoms=False, old_model=False, ns=8, nv=2, num_conv_layers=3,
        num_prot_emb_layers=1, sh_lmax=2, use_second_order_repr=False,   # the released value: node features stop at l=1
        reduce_pseudoscalars=True, embed_also_ligand=True, differentiate_convolutions=True,
        batch_norm=True, no_torsion=False, scale_by_sigma=True, odd_parity=False,
        tp_weights_layers=2, sigma_embed_dim=8, distance_embed_dim=8,
        cross_distance_embed_dim=8, embedding_type="sinusoidal", embedding_scale=1000,
        lig_max_radius=5.0, rec_max_radius=15.0, c_alpha_max_neighbors=24, atom_radius=5.0,
        atom_max_neighbors=8, cross_max_distance=80.0, dynamic_max_cross=True,
        center_max_distance=30.0, crop_beyond=20.0, tr_sigma_min=0.1, tr_sigma_max=19.0,
        rot_sigma_min=0.03, rot_sigma_max=1.55, tor_sigma_min=0.0314, tor_sigma_max=3.14)
    base.update(kw)
    return DiffDockConfig(**base)


def _model(**kw):
    torch.manual_seed(0)
    m = ScoreModel(_config(**kw), lm_embedding_dim=4).double().eval()
    for mod in m.modules():
        if hasattr(mod, "running_var") and isinstance(mod.running_var, torch.Tensor):
            with torch.no_grad():
                mod.running_var.uniform_(0.5, 2.0)
    return m


def _inputs(n_lig=9, n_rec=14):
    g = torch.Generator().manual_seed(3)
    # several of the 16 categorical tables have only two entries
    lig_x = torch.randint(0, 2, (n_lig, 16), generator=g)
    lig_pos = torch.randn(n_lig, 3, generator=g, dtype=torch.float64) * 2
    src = torch.arange(n_lig - 1)
    bond_index = torch.cat([torch.stack([src, src + 1]), torch.stack([src + 1, src])], dim=1)
    bond_attr = torch.zeros(bond_index.shape[1], 4, dtype=torch.float64)
    bond_attr[:, 0] = 1.0
    edge_mask = torch.zeros(bond_index.shape[1], dtype=torch.bool)
    edge_mask[2] = True
    rec_x = torch.cat([torch.randint(0, 2, (n_rec, 1), generator=g).double(),
                       torch.randn(n_rec, 4, generator=g, dtype=torch.float64)], dim=1)
    rec_pos = torch.randn(n_rec, 3, generator=g, dtype=torch.float64) * 6
    ri, rj = torch.triu_indices(n_rec, n_rec, offset=1)
    rec_edge_index = torch.cat([torch.stack([ri, rj]), torch.stack([rj, ri])], dim=1)
    return (lig_x, lig_pos, bond_index, bond_attr, edge_mask, rec_x, rec_pos, rec_edge_index)


SIGMAS = (torch.tensor(4.0), torch.tensor(0.9), torch.tensor(0.7))


def test_the_released_checkpoint_sets_differentiate_convolutions():
    """4 groups per conv layer and 2 on the last, which runs on ``[:s2]``. Loading a
    single-group assembly from this checkpoint silently loads nothing in those layers."""
    m = _model()
    counts = [len(lyr.fc) if isinstance(lyr.fc, torch.nn.ModuleList) else 1
              for lyr in m.conv_layers]
    assert counts == [4, 4, 2], counts
    # the embedding towers stay single-group: one FC block, not a ModuleList of them
    assert all(not isinstance(lyr.fc, torch.nn.ModuleList) for lyr in m.rec_emb_layers)


def test_the_head_regrouping_is_the_only_renaming():
    sd = {"center_edge_embedding.0.weight": 1, "tor_bond_conv.fc.0.weight": 2,
          "conv_layers.0.fc.0.0.weight": 3, "final_tp_tor._w3j_1_2_1": 4,
          "lig_emb_layers.0.tp.weight": 5}
    out = remap_checkpoint(sd)
    assert out == {"center_head.center_edge_embedding.0.weight": 1,
                   "torsion_head.tor_bond_conv.fc.0.weight": 2,
                   "conv_layers.0.fc.0.0.weight": 3}
    assert all(v.startswith(("center_head.", "torsion_head.")) for v in HEAD_PREFIXES.values())


def test_the_receptor_cache_is_the_same_answer_as_recomputing():
    """The receptor does not move while the ligand is denoised, so its tower is computed once
    -- but only when there is no crop, which changes the tower's input every step."""
    m = _model()
    args = _inputs()
    with torch.no_grad():
        a = m(*args, 0.5, *SIGMAS)
        cache = m.receptor_cache
        b = m(*args, 0.5, *SIGMAS, receptor_cache=cache)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_the_network_is_rotation_equivariant():
    """Rotate the whole complex: tr and rot rotate with it, the torsion score does not move.
    This is the campaign's accuracy gate, on the assembled network rather than a module."""
    m = _model()
    lig_x, lig_pos, bi, ba, em, rec_x, rec_pos, rei = _inputs()
    c, s = math.cos(0.8), math.sin(0.8)
    r = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=torch.float64)
    with torch.no_grad():
        tr, rot, tor = m(lig_x, lig_pos, bi, ba, em, rec_x, rec_pos, rei, 0.5, *SIGMAS)
        tr2, rot2, tor2 = m(lig_x, lig_pos @ r.T, bi, ba, em, rec_x, rec_pos @ r.T, rei,
                            0.5, *SIGMAS)
    for got, want in ((tr2, tr @ r.T), (rot2, rot @ r.T)):
        assert (got - want).abs().max() / want.abs().max() < 1e-10
    assert (tor2 - tor).abs().max() / tor.abs().max() < 1e-10


def test_radius_edges_takes_every_neighbour_not_upstreams_32():
    """Upstream's ``torch_cluster.radius_graph`` caps a node at 32 neighbours; this port does
    not, so where the cap binds the port sees a superset. Pinned so the difference is a
    decision, not a surprise."""
    pos = torch.zeros(40, 3, dtype=torch.float64)
    pos[:, 0] = torch.linspace(0, 0.5, 40)
    e = radius_edges(pos, 5.0)
    assert e.shape[1] == 40 * 39
    assert (e[0] != e[1]).all()


def test_radius_edges_is_invariant_to_atom_order():
    """The property that makes taking every neighbour the right call where upstream caps.

    ``torch_cluster.radius_graph``'s capped selection comes out of its cell-list traversal,
    so relabelling the same points changes which edges survive: measured, 16 of 1920 kept
    edges moved on a 60-point cloud whose true degrees were 57-59, and 0 of 60 nodes kept an
    index-order prefix. Upstream's score therefore depends on ligand atom order once the cap
    binds. This port's graph is a pure function of the geometry, and that is pinned here.
    """
    torch.manual_seed(0)
    pos = torch.randn(48, 3, dtype=torch.float64) * 0.8  # dense: every degree well over 32
    base = radius_edges(pos, 5.0)
    assert base.shape[1] > 48 * 32, "the cloud must be dense enough for upstream's cap to bind"

    perm = torch.randperm(48)
    moved = radius_edges(pos[perm], 5.0)
    remapped = torch.stack([perm[moved[0]], perm[moved[1]]])

    as_set = lambda e: {(int(a), int(b)) for a, b in zip(*e.tolist())}
    assert as_set(remapped) == as_set(base)


def test_atom_encoder_categorical_sum_hoists_out_of_the_step():
    """The encoder's categorical half is a per-complex constant, and hoisting it is exact.

    `AtomEncoder` sums one embedding table per categorical atom feature and then runs one
    linear over `cat([sum, continuous block])`. Atom types do not change over a diffusion
    rollout, so the whole sum can be evaluated once per complex and only the linear re-run
    per step -- which is what `device_loop.NodeEncoder` does, and why a step needs no gather
    on the card. Hoisting a loop invariant is not skipping the model's work: every step still
    evaluates the same linear over the same values.

    Checked bit-exact here rather than to a tolerance, because an embedding lookup is a copy
    and the concatenation order is the only thing that could differ.
    """
    import torch

    from tt_bio.diffdock.score_model import AtomEncoder

    torch.manual_seed(0)
    n_nodes = 37
    for feature_dims, sigma_dim, lm_dim in (([[5, 3, 9, 2], 0], 64, 0),
                                            ([[7], 1280], 0, 0),
                                            ([[4, 6], 3], 16, 0)):
        enc = AtomEncoder(48, feature_dims, sigma_dim, lm_embedding_dim=lm_dim).double()
        n_cat = enc.n_categorical
        cat_cols = torch.cat([torch.randint(0, emb.num_embeddings, (n_nodes, 1)).double()
                              for emb in enc.atom_embedding_list], 1)
        static_extra = enc.extra_dim - sigma_dim
        node_x = torch.cat([cat_cols, torch.randn(n_nodes, static_extra).double()], 1)
        sigma = torch.randn(1, sigma_dim).double().expand(n_nodes, -1) if sigma_dim else None

        reference = enc(node_x if sigma is None else torch.cat([node_x, sigma], 1))

        total = None
        for i, emb in enumerate(enc.atom_embedding_list):
            term = emb(node_x[:, i].long())
            total = term if total is None else total + term
        base = torch.cat([total, node_x[:, n_cat:]], 1)
        hoisted = enc.additional_features_embedder(
            base if sigma is None else torch.cat([base, sigma], 1))

        assert torch.equal(reference, hoisted), feature_dims
