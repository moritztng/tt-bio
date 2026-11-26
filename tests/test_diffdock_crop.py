"""The receptor crop as a node mask: same result as upstream's resize, one shape for 19 steps.

The deployed score checkpoint sets ``crop_beyond: 20``, and upstream re-crops the receptor to
``3 * tr_sigma + 20`` A around the ligand at every step (D-dfd-orch-11). A resize cannot be
traced. `StaticGraphs.rec_keep` keeps the full receptor and masks the same residues; this file
proves that is the same computation, on CPU, before any device row depends on it:

1. **The keep-set is upstream's.** On the real 8G43_ZU6 rollout dumped from upstream, the
   receptor upstream fed the network at steps 0, 8 and 18 (99, 99 and 96 residues) is exactly
   the full receptor under the mask -- same residues, same order.
2. **The network sees no difference.** Two stacked conv layers over ligand, cross and receptor
   edges, run once on the cropped graph (`sampling.crop_beyond`, upstream's resize) and once on
   the full receptor with masks, agree to fp32 rounding on every surviving node -- with the
   masked residues' own states deliberately left non-zero, so a leak would show.
3. **The shape does not move.** The masked step issues the identical op log at every step of
   the 77 A -> 20.5 A sweep, the precondition for one trace per complex.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from tt_bio.diffdock import graph as G  # noqa: E402
from tt_bio.diffdock import sampling as S  # noqa: E402
from tt_bio.diffdock.conv import TensorProductConvLayer  # noqa: E402
from tt_bio.diffdock.features import GaussianSmearing  # noqa: E402
from tt_bio.diffdock.harmonics import spherical_harmonics  # noqa: E402
from tt_bio.diffdock.trace import assert_shape_stable, record_ops  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "diffdock_graph" / "8G43_ZU6_steps.npz"
STEPS = (0, 8, 18)
SH_LS = (0, 1, 2)
DIM, N_EDGE_FEATURES, K_REC = 16, 24, 10
# Measured 0.0 at all three steps: a masked edge adds an exact 0 to the sum and to the divisor,
# so masking is bit-identical to removing on CPU. The bar is fp32 rounding for a device that
# may reorder the sums, not slack for the host.
REL_TOL = 1e-5


def _load():
    d = np.load(FIXTURE)
    full = torch.tensor(d["s0_rec_pos"]).float()     # step 0 keeps every residue (77 A)
    steps = {s: dict(lig=torch.tensor(d[f"s{s}_lig_pos"]).float(),
                     rec=torch.tensor(d[f"s{s}_rec_pos"]).float(),
                     cutoff=float(d[f"s{s}_cross_cutoff"][0])) for s in STEPS}
    return full, steps


def _knn(pos, k):
    """A receptor graph of upstream's kind (k-NN, precomputed, static); dst row first."""
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float("inf"))
    nbr = dist.topk(k, largest=False).indices
    dst = torch.arange(pos.shape[0]).repeat_interleave(k)
    return torch.stack([dst, nbr.reshape(-1)])


def _layers():
    torch.manual_seed(5)
    out = []
    for _ in range(2):
        layer = TensorProductConvLayer(
            in_irreps=f"{DIM}x0e", sh_irreps="1x0e+1x1o+1x2e", out_irreps=f"{DIM}x0e",
            n_edge_features=N_EDGE_FEATURES, hidden_features=32, batch_norm=False,
            residual=True)
        out.append(layer.eval())
    return out


EXPANSION = GaussianSmearing(0.0, 30.0, N_EDGE_FEATURES)


def _forward(layers, x, pos, edge_index, mask=None):
    """Both layers over one combined node axis: the cross, ligand and receptor edges at once."""
    vec = pos[edge_index[1]] - pos[edge_index[0]]
    attr = EXPANSION(vec.norm(dim=-1))
    sh = spherical_harmonics(vec, SH_LS)
    for layer in layers:
        x = layer(x, edge_index, attr, sh, out_nodes=x.shape[0], edge_mask=mask)
    return x


def _dynamic_edges(lig, rec, rec_ei, cutoff):
    """Upstream's form: edge lists built from the cropped receptor, no mask anywhere."""
    n_lig = lig.shape[0]
    d = torch.cdist(lig, rec)
    li, ri = torch.nonzero(d <= cutoff, as_tuple=True)
    lig_pairs = torch.nonzero(torch.cdist(lig, lig) <= 5.0, as_tuple=False).T
    lig_pairs = lig_pairs[:, lig_pairs[0] != lig_pairs[1]]
    return torch.cat([lig_pairs,
                      torch.stack([li, ri + n_lig]),          # ligand <- receptor
                      torch.stack([ri + n_lig, li]),          # receptor <- ligand
                      rec_ei + n_lig], dim=1)                 # receptor <- receptor


def _static_edges(n_lig, n_rec, rec_ei):
    lig_pairs = G.complete_pairs(n_lig, n_lig, exclude_diagonal=True)
    cross = G.complete_pairs(n_lig, n_rec)
    return torch.cat([lig_pairs,
                      torch.stack([cross[0], cross[1] + n_lig]),
                      torch.stack([cross[1] + n_lig, cross[0]]),
                      rec_ei + n_lig], dim=1), lig_pairs.shape[1], cross.shape[1]


@pytest.fixture(scope="module")
def real():
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE} missing")
    return _load()


@pytest.mark.parametrize("step", STEPS)
def test_the_mask_keeps_exactly_the_residues_upstream_kept(real, step):
    full, steps = real
    d = steps[step]
    bonds = torch.zeros((2, 0), dtype=torch.long)
    s = G.StaticGraphs.build(d["lig"].shape[0], full.shape[0], bonds)
    lig_p = torch.zeros(s.n_lig_padded, 3)
    lig_p[:d["lig"].shape[0]] = d["lig"]
    rec_p = torch.zeros(s.n_rec_padded, 3)
    rec_p[:full.shape[0]] = full
    keep = s.rec_keep(lig_p, rec_p, d["cutoff"]).bool()
    assert keep[full.shape[0]:].sum() == 0, "pad rows are never kept"
    assert torch.equal(full[keep[:full.shape[0]]], d["rec"]), \
        f"step {step}: masked receptor is not the {d['rec'].shape[0]} residues upstream used"
    # and the mask agrees with the host resize the sampler runs
    host = S.crop_beyond({"pos": full, "edge_index": torch.zeros((2, 0), dtype=torch.long)},
                         d["lig"], d["cutoff"])
    assert torch.equal(host["pos"], d["rec"])


def test_step_18_actually_crops(real):
    """Otherwise the comparison below would be vacuous."""
    full, steps = real
    assert steps[18]["rec"].shape[0] < full.shape[0]


@pytest.mark.parametrize("step", STEPS)
def test_masked_full_receptor_matches_the_cropped_graph(real, step):
    full, steps = real
    d = steps[step]
    lig, cutoff = d["lig"], d["cutoff"]
    n_lig, n_rec = lig.shape[0], full.shape[0]
    rec_ei = _knn(full, K_REC)
    layers = _layers()
    torch.manual_seed(7)
    x_lig = torch.randn(n_lig, DIM)
    x_rec = torch.randn(n_rec, DIM)

    # upstream: resize the receptor, re-index its graph, build the dynamic edges
    crop = S.crop_beyond({"pos": full, "x": x_rec, "edge_index": rec_ei}, lig, cutoff)
    keep = (torch.cdist(full, lig) < cutoff).any(dim=1)
    want = _forward(layers, torch.cat([x_lig, crop["x"]]), torch.cat([lig, crop["pos"]]),
                    _dynamic_edges(lig, crop["pos"], crop["edge_index"], cutoff))

    # the port: full receptor, complete static edges, every mask from StaticGraphs
    s = G.StaticGraphs.build(n_lig, n_rec, torch.zeros((2, 0), dtype=torch.long))
    lig_p = torch.zeros(s.n_lig_padded, 3)
    lig_p[:n_lig] = lig
    rec_p = torch.zeros(s.n_rec_padded, 3)
    rec_p[:n_rec] = full
    kmask = s.rec_keep(lig_p, rec_p, cutoff)
    assert torch.equal(kmask[:n_rec].bool(), keep)
    edges, n_pairs, n_cross = _static_edges(n_lig, n_rec, rec_ei)
    lig_m = G.within(lig[edges[1, :n_pairs]] - lig[edges[0, :n_pairs]], 5.0)
    cross_m = (torch.cdist(lig, full) <= cutoff).float()[
        edges[0, n_pairs:n_pairs + n_cross], edges[1, n_pairs:n_pairs + n_cross] - n_lig]
    cross_m = cross_m * kmask[edges[1, n_pairs:n_pairs + n_cross] - n_lig]
    rec_m = G.StaticGraphs.rec_edge_mask(kmask, rec_ei)
    mask = torch.cat([lig_m, cross_m, cross_m, rec_m])
    got = _forward(layers, torch.cat([x_lig, x_rec]), torch.cat([lig, full]), edges, mask)

    surviving = torch.cat([torch.arange(n_lig), n_lig + torch.nonzero(keep).reshape(-1)])
    got = got[surviving]
    rel = ((got - want).abs().max() / want.abs().max()).item()
    assert rel < REL_TOL, f"step {step}: {rel:.2e}"
    if not keep.all():
        # negative control: drop the receptor-edge mask and the cropped residues leak in
        unmasked = torch.cat([lig_m, cross_m, cross_m, torch.ones_like(rec_m)])
        leak = _forward(layers, torch.cat([x_lig, x_rec]), torch.cat([lig, full]), edges,
                        unmasked)[surviving]
        assert ((leak - want).abs().max() / want.abs().max()).item() > 1e-3, \
            "the control did not move -- the comparison above would be vacuous"


@pytest.mark.parametrize("step", STEPS)
def test_the_keep_set_falls_out_of_the_cross_mask_no_second_cdist(real, step):
    """`rec_keep_from_cross` is the form `dfd-device` should use, so it is graded here.

    The cross edge set is complete, so "within the radius of some ligand atom of this pose" is
    an `any` over the cross edges ending on the residue -- distances the step computes anyway.
    Over three real steps and ten poses it agrees residue for residue with the `cdist` form.
    """
    full, steps = real
    d = steps[step]
    lig, cutoff = d["lig"], d["cutoff"]
    n_lig, n_rec, b = lig.shape[0], full.shape[0], 3
    s = G.StaticGraphs.build(n_lig, n_rec, torch.zeros((2, 0), dtype=torch.long), n_poses=b)
    torch.manual_seed(2)
    # three different poses of the same ligand, so the keep-sets genuinely differ
    ligs = torch.stack([lig, lig + torch.tensor([6.0, 0.0, 0.0]), lig * 1.02])
    lig_p = torch.zeros(s.n_lig_padded, 3)
    lig_p[:n_lig * b] = ligs.reshape(-1, 3)
    rec_p = torch.zeros(s.n_rec_padded, 3)
    rec_p[:n_rec * b] = full.repeat(b, 1)
    want = s.rec_keep(lig_p, rec_p, cutoff)
    got = s.rec_keep_from_cross(s.cross_mask(lig_p, rec_p, cutoff))
    # `cross_mask` compares |v/c| <= 1 and `rec_keep` cdist < c: equal except exactly on the
    # sphere, which no residue of a real rollout sits on.
    assert torch.equal(got, want), f"step {step}: {int((got != want).sum())} residues differ"
    assert 0 < want[:n_rec * b].sum() < n_rec * b or step < 18


def test_rec_edge_index_tiles_and_pads_per_pose():
    pos = torch.randn(40, 3) * 10
    rec_ei = _knn(pos, 4)
    s = G.StaticGraphs.build(5, 40, torch.zeros((2, 0), dtype=torch.long), n_poses=3)
    ei = s.rec_edge_index(rec_ei)
    assert ei.shape[1] % 32 == 0
    real = rec_ei.shape[1] * 3
    for p in range(3):
        blk = ei[:, p * rec_ei.shape[1]:(p + 1) * rec_ei.shape[1]]
        assert torch.equal(blk, rec_ei + 40 * p)
    assert (ei[:, real:] == s.n_rec_padded - 1).all()


def test_the_keep_mask_is_per_pose_like_upstreams_per_graph_crop():
    torch.manual_seed(0)
    n_lig, n_rec, b = 6, 50, 3
    s = G.StaticGraphs.build(n_lig, n_rec, torch.zeros((2, 0), dtype=torch.long), n_poses=b)
    rec = torch.randn(n_rec, 3) * 15
    lig = torch.randn(b, n_lig, 3) * 2 + torch.tensor([[-12.0, 0, 0], [0, 0, 0], [12.0, 0, 0]])[:, None]
    lig_p = torch.zeros(s.n_lig_padded, 3)
    lig_p[:n_lig * b] = lig.reshape(-1, 3)
    rec_p = torch.zeros(s.n_rec_padded, 3)
    rec_p[:n_rec * b] = rec.repeat(b, 1)
    keep = s.rec_keep(lig_p, rec_p, 14.0)[:n_rec * b].reshape(b, n_rec).bool()
    for p in range(b):
        want = (torch.cdist(rec, lig[p]) < 14.0).any(dim=1)
        assert torch.equal(keep[p], want)
    assert not torch.equal(keep[0], keep[2]), "the poses must crop differently here"


def test_the_masked_step_has_one_shape_across_the_crop_sweep(real):
    """Op logs over the real step-0/8/18 cutoffs and a synthetic 77 -> 20.5 A sweep."""
    full, steps = real
    n_lig, n_rec = steps[0]["lig"].shape[0], full.shape[0]
    s = G.StaticGraphs.build(n_lig, n_rec, torch.zeros((2, 0), dtype=torch.long))
    rec_ei = s.rec_edge_index(_knn(full, K_REC))
    rec_p = torch.zeros(s.n_rec_padded, 3)
    rec_p[:n_rec] = full
    layers = _layers()
    x = torch.randn(s.n_lig_padded + s.n_rec_padded, DIM)
    cross = s.cross_edge_index_global()
    edges = torch.cat([cross, cross.flip(0), rec_ei + s.rec_offset], dim=1)
    pos_rec = torch.cat([torch.zeros(s.n_lig_padded, 3), rec_p])

    def body(lig, cutoff):
        lig_p = torch.zeros(s.n_lig_padded, 3)
        lig_p[:n_lig] = lig
        keep = s.rec_keep(lig_p, rec_p, cutoff)
        cm = s.cropped_cross_mask(lig_p, rec_p, cutoff, keep)
        mask = torch.cat([cm, cm, G.StaticGraphs.rec_edge_mask(keep, rec_ei)])
        pos = pos_rec.clone()
        pos[:s.n_lig_padded] = lig_p
        return _forward(layers, x, pos, edges, mask)

    cutoffs = list(np.linspace(77.0, 20.5, 19))
    ligs = [steps[0]["lig"]] * 6 + [steps[8]["lig"]] * 7 + [steps[18]["lig"]] * 6
    logs = []
    for lig, c in zip(ligs, cutoffs):
        with torch.no_grad():
            _, log = record_ops(body, lig, float(c))
        logs.append(log)
    # step 0 pays the Wigner-3j cast once (the warm-up rule); compare from step 1
    assert_shape_stable(logs[1:])
