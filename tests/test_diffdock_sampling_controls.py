"""The reverse-diffusion sampler: its controls, and its trajectory against upstream.

The rollout goldens in ``tests/fixtures/diffdock/sampling_rollout_golden.npz`` are upstream's
own ``utils.sampling.sampling`` run on a stub score network, captured by
``scripts/diffdock_port/sampling_parity.py --write-golden`` in the reference environment. The
stub is a deterministic function of the pose and of t, so a wrong schedule, a wrong dt on the
last step or a draw consumed out of order moves the trajectory; what is graded here is the
rollout, which is what this module owns.

Entry k of a golden is the end state of a k-step run, not step k of one 19-step run: the last
step of any run takes dt to t=0 and skips its noise, so a short run is not a prefix of a long
one. Both sides are read out the same way.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock import sampling                                  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "diffdock" / "sampling_rollout_golden.npz"
BOUNDS = sampling.SigmaBounds(0.1, 19.0, 0.03, 1.55, 0.0314, 3.14)


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"{GOLDEN.name} not captured yet")
    return np.load(GOLDEN)


def _stub_scores(pos, t_tr, t_rot, t_tor, n_torsions):
    """The same stub the golden was captured with (sampling_parity.StubModel)."""
    centre = pos.mean(dim=1)
    tr = torch.tanh(centre / 10.0) * (1.0 + t_tr)
    rot = torch.sin(pos[:, 0] - pos[:, -1]) * (1.0 + t_rot)
    tor = torch.cos(pos[:, :n_torsions, 0] + t_tor).reshape(-1)
    return tr, rot, tor


def _run_golden_case(g, cfg, *, seed=None, every_k=True):
    pos = torch.from_numpy(g["pos"])
    edge_index = torch.from_numpy(g["edge_index"])
    edge_mask = torch.from_numpy(g["edge_mask"])
    mask_rotate = g["mask_rotate"]
    n_poses = g["sde_trajectory"].shape[1]
    start = pos.unsqueeze(0).repeat(n_poses, 1, 1)

    def score_fn(step):
        return _stub_scores(step.lig_pos, step.t_tr, step.t_rot, step.t_tor,
                            mask_rotate.shape[0])

    def run(steps):
        # Unseeded against a golden: upstream has no seed, so the global streams are set to
        # the value the capture used. The extra int64 draw is upstream's DataLoader base
        # seed, which it consumes before the first step; the port has no loader, so the
        # streams have to be lined up here. Seeded: the sampler's own --seed does it.
        if seed is None:
            torch.manual_seed(int(g["seed"]))
            np.random.seed(int(g["seed"]))
            torch.empty((), dtype=torch.int64).random_()
        return sampling.sample(score_fn, start, cfg=replace(cfg, actual_steps=steps),
                               bounds=BOUNDS, noise=sampling.Noise(seed=seed),
                               edge_index=edge_index.T[edge_mask],
                               mask_rotate=mask_rotate).poses

    if not every_k:
        return run(cfg.steps)
    return [run(k) for k in range(1, cfg.steps + 1)]


def test_full_rollout_matches_cpu_golden(golden):
    """Every step of the default-recipe rollout, against upstream's own loop."""
    cfg = sampling.SamplingConfig.from_recipe()
    out = _run_golden_case(golden, cfg)
    ref = torch.from_numpy(golden["sde_trajectory"])
    assert len(out) == ref.shape[0] == cfg.steps
    worst = max(float((ref[k] - out[k]).abs().max()) for k in range(len(ref)))
    assert worst < 1e-5, f"worst deviation {worst:.3e} A"


def test_ode_path_matches_golden(golden):
    """The probability-flow variant: no noise term, same schedule, same conformer update."""
    cfg = sampling.SamplingConfig.from_recipe(ode=True, temp_sampling=(1.0, 1.0, 1.0),
                                              temp_psi=(0.0, 0.0, 0.0))
    out = _run_golden_case(golden, cfg)
    ref = torch.from_numpy(golden["ode_trajectory"])
    worst = max(float((ref[k] - out[k]).abs().max()) for k in range(len(ref)))
    assert worst < 1e-5, f"worst deviation {worst:.3e} A"


def test_ode_refuses_the_low_temperature_constants():
    """Upstream's ODE branch draws no noise for the temperature terms to reweight."""
    cfg = sampling.SamplingConfig.from_recipe(ode=True)
    with pytest.raises(ValueError, match="ode"):
        sampling.sample(lambda step: None, torch.zeros(2, 3, 3), cfg=cfg, bounds=BOUNDS)


def test_same_seed_same_poses(golden):
    """``--seed`` is reproducible pose for pose, and two seeds genuinely differ."""
    cfg = sampling.SamplingConfig.from_recipe()
    a = _run_golden_case(golden, cfg, seed=4242, every_k=False)
    b = _run_golden_case(golden, cfg, seed=4242, every_k=False)
    c = _run_golden_case(golden, cfg, seed=99, every_k=False)
    assert torch.equal(a, b)
    assert float((a - c).abs().max()) > 1e-3


def test_seeded_initial_poses_are_reproducible():
    """The initial randomisation is seeded too: three generators, one seed."""
    pos, receptor = torch.randn(9, 3), torch.randn(30, 3) * 8
    cfg = sampling.SamplingConfig.from_recipe()
    kw = dict(receptor_pos=receptor, n_samples=3, cfg=cfg, bounds=BOUNDS, no_torsion=True)
    a = sampling.randomize_position(pos, noise=sampling.Noise(seed=1), **kw)
    b = sampling.randomize_position(pos, noise=sampling.Noise(seed=1), **kw)
    c = sampling.randomize_position(pos, noise=sampling.Noise(seed=2), **kw)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_no_random_is_deterministic():
    """``--no_random`` drops every stochastic term: the rollout is a function of the start."""
    cfg = sampling.SamplingConfig.from_recipe(no_random=True, temp_sampling=(1.0, 1.0, 1.0),
                                              temp_psi=(0.0, 0.0, 0.0))
    start = torch.randn(3, 6, 3)

    def score_fn(step):
        return (torch.ones(3, 3) * 0.1, torch.ones(3, 3) * 0.05, None)

    a = sampling.sample(score_fn, start, cfg=cfg, bounds=BOUNDS, noise=sampling.Noise(),
                        no_torsion=True).poses
    b = sampling.sample(score_fn, start, cfg=cfg, bounds=BOUNDS, noise=sampling.Noise(),
                        no_torsion=True).poses
    assert torch.equal(a, b)


def test_steps_and_actual_steps_honoured():
    """19 of 20: the schedule keeps its 20 points and the last step's dt jumps to t=0."""
    cfg = sampling.SamplingConfig.from_recipe()
    assert (cfg.inference_steps, cfg.actual_steps, cfg.steps) == (20, 19, 19)
    assert len(cfg.schedules(BOUNDS)[0]) == 20

    seen = []

    def score_fn(step):
        seen.append((step.step, step.t_tr, step.tr_sigma))
        return (torch.zeros(2, 3), torch.zeros(2, 3), None)

    sampling.sample(score_fn, torch.randn(2, 5, 3), cfg=cfg, bounds=BOUNDS, no_torsion=True,
                    noise=sampling.Noise(seed=0))
    assert len(seen) == 19
    sched = cfg.schedules(BOUNDS)[0]
    assert seen[-1][1] == pytest.approx(float(sched[18]))
    assert sampling.SamplingConfig.from_recipe(actual_steps=None).steps == 20


def test_sample_count_poses_written():
    """``samples_per_complex`` poses go in and the same number comes out, pose for pose."""
    cfg = sampling.SamplingConfig.from_recipe()
    assert cfg.samples_per_complex == 10
    start = sampling.randomize_position(
        torch.randn(7, 3), receptor_pos=torch.randn(25, 3) * 6,
        n_samples=cfg.samples_per_complex, cfg=cfg, bounds=BOUNDS,
        noise=sampling.Noise(seed=5), no_torsion=True)
    assert start.shape == (10, 7, 3)
    out = sampling.sample(lambda s: (torch.zeros(10, 3), torch.zeros(10, 3), None), start,
                          cfg=cfg, bounds=BOUNDS, no_torsion=True, noise=sampling.Noise(seed=5))
    assert out.poses.shape == (10, 7, 3)


def test_final_step_noise_flag():
    """With the flag on, the last step injects no noise; with it off, it does."""
    start = torch.randn(4, 5, 3)
    score_fn = lambda step: (torch.zeros(4, 3), torch.zeros(4, 3), None)   # noqa: E731
    kw = dict(bounds=BOUNDS, no_torsion=True)

    def last_step_delta(no_final_step_noise):
        cfg = sampling.SamplingConfig.from_recipe(
            no_final_step_noise=no_final_step_noise, inference_steps=3, actual_steps=2)
        out = sampling.sample(score_fn, start, cfg=cfg, noise=sampling.Noise(seed=1),
                              trajectory=True, **kw)
        return float((out.trajectory[-1] - out.trajectory[-2]).abs().max())

    # A zero score means the only thing that can move the pose on the last step is the noise.
    assert last_step_delta(True) < 1e-6
    assert last_step_delta(False) > 1e-3


def test_low_temperature_constants_applied():
    """The nine fitted constants change the step, and they change it by the fitted amount."""
    cfg = sampling.SamplingConfig.from_recipe()
    assert cfg.temp_sampling == pytest.approx((1.170050527854316, 2.06391612594481,
                                               7.044261621607846))
    tr_sigma = 19.0
    factor, width = sampling._temperature_scale(0, cfg, tr_sigma, BOUNDS, "tr")
    sd = np.exp(cfg.temp_sigma_data[0] * np.log(BOUNDS.tr_sigma_max)
                + (1 - cfg.temp_sigma_data[0]) * np.log(BOUNDS.tr_sigma_min))
    lam = (sd + tr_sigma) / (sd + tr_sigma / cfg.temp_sampling[0])
    assert factor == pytest.approx(lam + cfg.temp_sampling[0] * cfg.temp_psi[0] / 2)
    assert width == pytest.approx(1 + cfg.temp_psi[0])

    off = sampling.SamplingConfig.from_recipe(temp_sampling=(1.0, 1.0, 1.0))
    assert sampling._temperature_scale(0, off, tr_sigma, BOUNDS, "tr") == (1.0, 1.0)

    start = torch.randn(2, 4, 3)
    score_fn = lambda step: (torch.ones(2, 3) * 0.2, torch.zeros(2, 3), None)  # noqa: E731
    kw = dict(bounds=BOUNDS, no_torsion=True, no_random=True)
    hot = sampling.sample(score_fn, start, cfg=sampling.SamplingConfig.from_recipe(
        no_random=True), noise=sampling.Noise(seed=2), **{k: v for k, v in kw.items()
                                                         if k != "no_random"}).poses
    cold = sampling.sample(score_fn, start, cfg=sampling.SamplingConfig.from_recipe(
        no_random=True, temp_sampling=(1.0, 1.0, 1.0), temp_psi=(0.0, 0.0, 0.0)),
        noise=sampling.Noise(seed=2), **{k: v for k, v in kw.items() if k != "no_random"}).poses
    assert float((hot - cold).abs().max()) > 1e-2


def test_initial_noise_proportion():
    """1.4601642460337794 scales the receptor's RMS radius; a negative value scales sigma_max."""
    cfg = sampling.SamplingConfig.from_recipe()
    assert cfg.initial_noise_std_proportion == pytest.approx(1.4601642460337794)
    receptor = torch.randn(400, 3) * 11.0
    pos = torch.randn(6, 3)
    kw = dict(receptor_pos=receptor, n_samples=400, cfg=cfg, bounds=BOUNDS, no_torsion=True)
    wide = sampling.randomize_position(pos, noise=sampling.Noise(seed=0), **kw)
    std_rec = float(torch.sqrt(torch.mean(torch.sum(receptor ** 2, dim=1))))
    expect = std_rec * cfg.initial_noise_std_proportion / 1.73
    centres = wide.mean(dim=1) - receptor.mean(dim=0)
    assert float(centres.std()) == pytest.approx(expect, rel=0.15)

    tight = sampling.randomize_position(
        pos, noise=sampling.Noise(seed=0),
        **{**kw, "cfg": sampling.SamplingConfig.from_recipe(initial_noise_std_proportion=-0.1)})
    assert float((tight.mean(dim=1) - receptor.mean(dim=0)).std()) == pytest.approx(1.9, rel=0.2)


def test_choose_residue_init():
    """``--choose_residue`` starts each pose on a receptor residue, not on the centroid."""
    # Centred, as every graph out of `input.build_complex_graph` is: upstream *adds* the
    # chosen residue's position to a pose already placed at the receptor centre, so the pose
    # lands on the residue only in the centred frame the pipeline actually uses.
    receptor = torch.randn(50, 3) * 20.0
    receptor = receptor - receptor.mean(dim=0)
    cfg = sampling.SamplingConfig.from_recipe(choose_residue=True)
    poses = sampling.randomize_position(torch.randn(5, 3), receptor_pos=receptor, n_samples=8,
                                        cfg=cfg, bounds=BOUNDS, noise=sampling.Noise(seed=3),
                                        no_torsion=True)
    centres = poses.mean(dim=1)
    nearest = torch.cdist(centres, receptor).min(dim=1).values
    assert float(nearest.max()) < 0.2          # within the 0.01 A jitter of a real residue
    assert float(centres.std(dim=0).mean()) > 1.0


def test_pocket_conditioned_init():
    """Pocket knowledge starts the pose in the pocket and with the pocket's sigma_max."""
    torch.manual_seed(0)
    pocket = torch.tensor([[30.0, 0.0, 0.0]]) + torch.randn(6, 3) * 0.5
    receptor = torch.cat([pocket, torch.randn(200, 3) * 8.0])     # a pocket off to one side
    reference = pocket.mean(dim=0, keepdim=True)                  # a "known" ligand in it
    centre = sampling.pocket_centre(receptor, reference, cutoff=5.0)
    assert float((centre - pocket.mean(dim=0)).abs().max()) < 1.0

    cfg = sampling.SamplingConfig.from_recipe(pocket_knowledge=True, pocket_tr_max=3.0,
                                              initial_noise_std_proportion=-1.0)
    poses = sampling.randomize_position(torch.randn(5, 3), receptor_pos=receptor, n_samples=64,
                                        cfg=cfg, bounds=BOUNDS, noise=sampling.Noise(seed=6),
                                        no_torsion=True, centre=centre)
    spread = float((poses.mean(dim=1) - centre).std())
    assert spread == pytest.approx(3.0, rel=0.3)         # pocket_tr_max, not tr_sigma_max
    blind = sampling.randomize_position(
        torch.randn(5, 3), receptor_pos=receptor, n_samples=64,
        cfg=sampling.SamplingConfig.from_recipe(initial_noise_std_proportion=-1.0),
        bounds=BOUNDS, noise=sampling.Noise(seed=6), no_torsion=True)
    assert float((blind.mean(dim=1) - receptor.mean(dim=0)).std()) > 10.0


def test_crop_beyond_node_counts_match():
    """The per-step crop keeps exactly the residues within 3*tr_sigma + 20 of the pose."""
    torch.manual_seed(0)
    receptor = {
        "pos": torch.randn(120, 3) * 25.0,
        "x": torch.randn(120, 7),
        "side_chain_vecs": torch.randn(120, 4, 3),
        "edge_index": torch.randint(0, 120, (2, 400)),
    }
    lig = torch.randn(9, 3) * 2.0
    cutoff = 3 * 0.5 + 20.0
    cropped = sampling.crop_beyond(receptor, lig, cutoff)

    keep = torch.any(torch.cdist(receptor["pos"], lig) < cutoff, dim=1)
    assert cropped["pos"].shape[0] == int(keep.sum()) < 120
    assert cropped["x"].shape[0] == cropped["side_chain_vecs"].shape[0] == int(keep.sum())
    # Every surviving edge joins two kept residues, re-indexed into the cropped numbering.
    assert int(cropped["edge_index"].max()) < int(keep.sum())
    both = keep[receptor["edge_index"][0]] & keep[receptor["edge_index"][1]]
    assert cropped["edge_index"].shape[1] == int(both.sum())
    assert receptor["pos"].shape[0] == 120                   # the input is not mutated


def test_crop_cutoff_shrinks_with_the_schedule():
    """The cutoff handed to the score network anneals with tr_sigma, as upstream's does."""
    cfg = sampling.SamplingConfig.from_recipe()
    cutoffs = []

    def score_fn(step):
        cutoffs.append(step.crop_cutoff)
        return (torch.zeros(2, 3), torch.zeros(2, 3), None)

    sampling.sample(score_fn, torch.randn(2, 4, 3), cfg=cfg, bounds=BOUNDS, no_torsion=True,
                    noise=sampling.Noise(seed=0), crop_beyond_a=20.0)
    assert cutoffs[0] == pytest.approx(3 * 19.0 + 20.0)
    assert cutoffs[-1] < cutoffs[0] and cutoffs[-1] > 20.0
    assert all(a >= b for a, b in zip(cutoffs, cutoffs[1:]))


def test_torsion_update_preserves_bond_lengths():
    """A torsion move rotates about a bond: every bond length survives it."""
    pos = torch.randn(2, 10, 3)
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, 5:] = True
    edges = torch.tensor([[4, 5]])
    out = sampling.torsion_update_batch(pos, edges, mask, torch.tensor([[0.7], [-1.2]]))
    before, after = torch.cdist(pos[0], pos[0]), torch.cdist(out[0], out[0])
    assert float((before[:5, :5] - after[:5, :5]).abs().max()) < 1e-5     # fixed side rigid
    assert float((before[5:, 5:] - after[5:, 5:]).abs().max()) < 1e-5     # moving side rigid
    assert float((before[4, 5] - after[4, 5]).abs()) < 1e-5               # the bond itself
    assert float((before - after).abs().max()) > 1e-2                     # but it did move


def test_axis_angle_matches_a_rotation():
    """The quaternion composition is a rotation, and it is the right one."""
    axis_angle = torch.tensor([[0.0, 0.0, np.pi / 2], [1e-9, 0.0, 0.0]])
    r = sampling.axis_angle_to_matrix(axis_angle)
    assert float((r[0] @ torch.tensor([1.0, 0.0, 0.0]) - torch.tensor([0.0, 1.0, 0.0]))
                 .abs().max()) < 1e-6
    assert float((r[1] - torch.eye(3)).abs().max()) < 1e-8        # small-angle branch
    assert float((torch.linalg.det(r) - 1).abs().max()) < 1e-6


def test_kabsch_batch_agrees_with_the_shipped_fp64_implementation():
    """The batched fp32 Kabsch in the loop against ``tt_bio.align``'s fp64 single-pair one."""
    from tt_bio import align

    a = torch.randn(3, 20, 3)
    rot = sampling.axis_angle_to_matrix(torch.randn(3, 3))
    b = torch.bmm(a, rot.transpose(1, 2)) + torch.randn(3, 1, 3)
    r, t = sampling.kabsch_batch(a, b)
    fitted = torch.bmm(a, r.transpose(1, 2)) + t.transpose(1, 2)
    assert float((fitted - b).abs().max()) < 1e-4
    for i in range(3):
        ref_r, mean_a, mean_b = align.rigid_transform(a[i], b[i])
        ref = (a[i].double() - mean_a) @ ref_r + mean_b
        assert float((ref.float() - fitted[i]).abs().max()) < 1e-4
