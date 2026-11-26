"""The 19-step loop issues the same program every step -- the precondition for trace capture.

This is the CPU half of deliverable 3. `ttnn.begin_trace_capture` records one pass and replays
it; a step whose op sequence or tensor shapes depend on the data cannot be replayed correctly.
Rather than argue that from the source, run the loop and compare the op logs step by step
(`tt_bio.diffdock.trace`). The body here is the real one: static graphs from
`StaticGraphs.build`, masks recomputed from the moving coordinates with the step's own cross
cutoff, a two-group convolution through `tp_scatter(edge_mask=...)`, the coefficient table,
the branch-free Kabsch and the masked torsion loop. Only the score *weights* are synthetic.

The cross cutoff shrinks from 77 A to 20.5 A over the 19 steps (`state/dfd/dfd-edges.json`),
which is exactly the thing that made upstream's graph dynamic: at step 0 every ligand-receptor
pair is inside it and at step 18 only 69.5 % are. If any shape in this loop followed the data,
that is the sweep that would expose it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock import graph as G  # noqa: E402
from tt_bio.diffdock.conv import TensorProductConvLayer  # noqa: E402
from tt_bio.diffdock.features import GaussianSmearing  # noqa: E402
from tt_bio.diffdock.geometry import spherical_harmonics  # noqa: E402
from tt_bio.diffdock.loop import build_schedule  # noqa: E402
from tt_bio.diffdock.conformer import modify_conformer  # noqa: E402
from tt_bio.diffdock.trace import (assert_shape_stable, first_difference,  # noqa: E402
                                    record_ops)

N_STEPS = 19
N_POSES = 4            # the shapes are per-pose-blocked; 4 proves the blocking, 10 is slower
N_LIG, N_REC, N_ROT = 24, 48, 6
LIG_MAX_RADIUS, CROSS_MAX_DISTANCE = 5.0, 80.0
SH_LS = (0, 1, 2)
SIGMAS = {'tr': (0.1, 19.0), 'rot': (0.03, 1.55), 'tor': (0.0314, 3.14)}


def _complex(seed=0):
    torch.manual_seed(seed)
    bonds = [(k, k + 1) for k in range(N_ROT)]
    lig_bond_edges = torch.tensor([[u for u, _ in bonds] + [v for _, v in bonds],
                                   [v for _, v in bonds] + [u for u, _ in bonds]])
    s = G.StaticGraphs.build(N_LIG, N_REC, lig_bond_edges, n_rot_bonds=N_ROT, n_poses=N_POSES)
    lig_pos = torch.randn(N_POSES * s.n_lig_padded, 3) * 3.0
    rec_pos = torch.randn(N_POSES * s.n_rec_padded, 3) * 12.0
    mask_rotate = torch.zeros(N_ROT, s.n_lig_padded, dtype=torch.bool)
    for k in range(N_ROT):
        mask_rotate[k, k + 1:N_LIG] = True
    return s, lig_pos, rec_pos, bonds, mask_rotate


def _step_fn(s, rec_pos, bonds, mask_rotate, table):
    """One loop body, closing over everything that is a loop invariant."""
    in_dim, n_edge_features = 16, 24
    torch.manual_seed(1)
    layer = TensorProductConvLayer(
        in_irreps=f"{in_dim}x0e", sh_irreps="1x0e+1x1o+1x2e",
        out_irreps=f"{in_dim}x0e+4x1o", n_edge_features=n_edge_features, edge_groups=2,
        hidden_features=32, batch_norm=False, residual=False)
    layer.eval()
    node_attr = torch.randn(N_POSES * (s.n_lig_padded + s.n_rec_padded), in_dim) * 0.1
    lig_expansion = GaussianSmearing(0.0, LIG_MAX_RADIUS, n_edge_features)
    cross_expansion = GaussianSmearing(0.0, CROSS_MAX_DISTANCE, n_edge_features)
    lig_ei, cross_ei = s.lig_edge_index, s.cross_edge_index_global()
    n_nodes = N_POSES * (s.n_lig_padded + s.n_rec_padded)

    def step(lig_pos, t_idx, cutoff, noise):
        lig_vec = lig_pos[lig_ei[1]] - lig_pos[lig_ei[0]]
        cross_vec = rec_pos[cross_ei[1] - s.rec_offset] - lig_pos[cross_ei[0]]
        edge_attr = [lig_expansion(lig_vec.norm(dim=-1)),
                     cross_expansion(cross_vec.norm(dim=-1))]
        edge_sh = torch.cat([spherical_harmonics(SH_LS, lig_vec),
                             spherical_harmonics(SH_LS, cross_vec)])
        mask = torch.cat([s.lig_mask(lig_pos, LIG_MAX_RADIUS),
                          s.cross_mask(lig_pos, rec_pos, cutoff)])
        out = layer(node_attr, torch.cat([lig_ei, cross_ei], dim=1), edge_attr, edge_sh,
                    out_nodes=n_nodes, edge_mask=mask)
        # a stand-in for the output heads: pool each pose's ligand block into the three scores
        pooled = out[:N_POSES * s.n_lig_padded].reshape(N_POSES, s.n_lig_padded, -1).mean(1)
        tr_score, rot_score = pooled[:, :3], pooled[:, 3:6]
        tor_score = pooled[:, 6:7].expand(N_POSES, N_ROT) * 0.1
        tr_z, rot_z, tor_z = noise
        sc, nz = table.row(t_idx)
        pos = lig_pos.reshape(N_POSES, s.n_lig_padded, 3)
        pos, _ = modify_conformer(pos, sc[0] * tr_score + nz[0] * tr_z,
                                  sc[1] * rot_score + nz[1] * rot_z,
                                  sc[2] * tor_score + nz[2] * tor_z, bonds, mask_rotate)
        return pos.reshape(-1, 3)

    return step


def _run(step, lig_pos, sched, n_steps, seed=5):
    torch.manual_seed(seed)
    logs = []
    for t_idx in range(n_steps):
        # upstream's own cutoff sweep: 3 * tr_sigma + 20, 77 A down to 20.5 A
        t = sched[t_idx]
        tr_sigma = SIGMAS['tr'][0] ** (1 - t) * SIGMAS['tr'][1] ** t
        noise = (torch.randn(N_POSES, 3), torch.randn(N_POSES, 3), torch.randn(N_POSES, N_ROT))
        lig_pos, log = record_ops(step, lig_pos, t_idx, 3 * tr_sigma + 20.0, noise)
        logs.append(log)
    return lig_pos, logs


def _loop():
    s, lig_pos, rec_pos, bonds, mask_rotate = _complex()
    sched = [1.0 - i * 0.05 for i in range(N_STEPS)]
    table = build_schedule(sched, sched, sched, SIGMAS)
    return _step_fn(s, rec_pos, bonds, mask_rotate, table), lig_pos, sched


def test_every_one_of_the_19_steps_is_the_same_program_after_a_warm_up():
    """The deliverable: one warm-up step, then 19 identical programs.

    Identical means op for op, shape for shape, dtype for dtype -- so the cross graph's 77 A
    to 20.5 A cutoff sweep, which is what made upstream's graph dynamic, moves only the
    *contents* of the mask.
    """
    step, lig_pos, sched = _loop()
    lig_pos, _ = _run(step, lig_pos, sched, 1)              # warm-up, see the test below
    lig_pos, logs = _run(step, lig_pos, sched, N_STEPS)

    n_ops = assert_shape_stable(logs)
    assert n_ops > 200, f"only {n_ops} ops recorded -- the body did not run"
    assert torch.isfinite(lig_pos).all(), "19 steps without a NaN"


def test_the_first_step_is_not_the_same_program_and_that_is_the_warm_up_rule():
    """Why the capture cannot be taken on the first step -- found by this harness, not guessed.

    `TensorProduct` keeps its Wigner 3j tables in float64 and casts them to the working dtype
    and device on first use (`tensor_product.py::_w3j`, `_w3j_cast`). That cast is six extra
    ops in step 0 and none afterwards. On a card it would allocate a device buffer *inside*
    `begin_trace_capture`, which is exactly what a capture cannot own. So the device pass runs
    the body once eagerly and captures the second pass -- and the same goes for any other
    lazily materialised constant added later, which this test will catch the same way.
    """
    step, lig_pos, sched = _loop()
    _, logs = _run(step, lig_pos, sched, 3)
    assert first_difference(logs[1], logs[2]) is None, "steps 1 and 2 must already agree"
    diff = first_difference(logs[0], logs[1])
    assert diff is not None, "if step 0 ever matches, drop the warm-up from the test above"
    assert len(logs[0]) > len(logs[1]), "step 0 is the one with the extra casts"


def test_the_recorder_catches_a_data_dependent_shape():
    """The test above is only worth something if this one fails the way it should."""
    def dynamic(x):
        return x[x.norm(dim=-1) < 1.0].sum()

    torch.manual_seed(0)
    logs = [record_ops(dynamic, torch.randn(64, 3))[1] for _ in range(3)]
    with pytest.raises(AssertionError, match="not the same program"):
        assert_shape_stable(logs)


def test_the_cutoff_sweep_really_does_change_the_mask():
    """Shape stability is only interesting because the *contents* move."""
    s, lig_pos, rec_pos, _, _ = _complex()
    wide = s.cross_mask(lig_pos, rec_pos, 77.0)
    tight = s.cross_mask(lig_pos, rec_pos, 20.5)
    assert wide.shape == tight.shape
    assert wide.sum() > tight.sum() * 1.2, "the sweep should drop a visible share of the edges"
