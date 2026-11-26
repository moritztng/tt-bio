"""The traced loop against `tt_bio.diffdock.sampling`, which is the graded sampler.

`sampling.sample` is the campaign's reference for the rollout: it is bit-identical to
upstream's own loop over 133 rollouts (dfd-sampling). So the bar for `loop.py` is not upstream
directly -- it is sampling, run on the same scores and the same noise draws. Where the two
differ by construction, the test says which difference and why:

* the coefficient table is float64 numpy arithmetic evaluated up front, sampling's is the
  same arithmetic evaluated inline, so the products land within fp32 rounding of each other;
* the Kabsch is Horn's quaternion rather than an SVD, whose measured float32 floor is ~1e-06
  on the rotation entries (`tests/test_diffdock_kabsch.py`), and that is what sets the
  tolerance on a rollout's final coordinates.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from tt_bio.diffdock import sampling as S  # noqa: E402
from tt_bio.diffdock.loop import (CoefficientTable, coefficient_table, guard_scores,  # noqa: E402
                                  run_loop)

BOUNDS = S.SigmaBounds(0.1, 19.0, 0.03, 1.55, 0.0314, 3.14)
# What "the same rollout" is worth measuring in: a relative bound on the final coordinates.
# In float64 the two agree to 2e-12 relative (test_in_float64_the_two_rollouts_are_the_same_
# function), so the fp32 gap below is rounding amplified over 19 steps -- Horn's quaternion
# sits ~1e-06 from the SVD on the rotation entries and each step's lever arm is tens of
# angstroms -- and not a difference in the arithmetic.
POSE_REL_TOL = 5e-5
POSE_REL_TOL_F64 = 1e-9


class RecordingNoise(S.Noise):
    """Sampling's RNG, with every per-step draw kept so the traced loop can replay it."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.draws = []

    def normal(self, *, mean=0.0, std=1.0, size=None):
        out = super().normal(mean=mean, std=std, size=size)
        self.draws.append(out)
        return out


def chain_ligand(n_atoms=20, n_poses=4, seed=0):
    """A chain with every bond rotatable: bond k is (k, k+1), rotating k+1 and beyond.

    A tree, so the mask is legal (sampling asserts `v` is on the moving side), and the worst
    case for the serial torsion loop -- every bond's axis has been moved by every earlier one.
    """
    g = torch.Generator().manual_seed(seed)
    pos = torch.cumsum(torch.randn(n_poses, n_atoms, 3, generator=g), dim=1) * 1.5
    bonds = [(k, k + 1) for k in range(n_atoms - 1)]
    mask = torch.zeros(len(bonds), n_atoms, dtype=torch.bool)
    for k in range(len(bonds)):
        mask[k, k + 1:] = True
    return pos, bonds, mask, g


def scripted_scores(n_steps, n_poses, n_bonds, seed=3):
    """Fixed per-step scores, so the two rollouts are compared on identical model output."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(n_poses, 3, generator=g) * 0.2,
             torch.randn(n_poses, 3, generator=g) * 0.2,
             torch.randn(n_poses * n_bonds, generator=g) * 0.2) for _ in range(n_steps)]


def rel_dev(a, b):
    """Max deviation relative to the coordinate scale -- a pose travels tens of angstroms."""
    return ((a - b).abs().max() / b.abs().max()).item()


def run_both(cfg):
    pos, bonds, mask, _ = chain_ligand()
    n_poses, n_bonds = pos.shape[0], len(bonds)
    scores = scripted_scores(cfg.steps, n_poses, n_bonds)
    edge_index = torch.tensor(bonds)

    noise = RecordingNoise(seed=11)
    eager = S.sample(lambda step: scores[step.step], pos, cfg=cfg, bounds=BOUNDS,
                     noise=noise, edge_index=edge_index, mask_rotate=mask)

    table = coefficient_table(cfg, BOUNDS)
    drawn = iter(noise.draws)

    def noise_fn(t_idx):
        if not table.drawn[t_idx]:
            z = (torch.zeros(n_poses, 3), torch.zeros(n_poses, 3),
                 torch.zeros(n_poses, n_bonds))
            return z
        return (next(drawn), next(drawn), next(drawn).reshape(n_poses, n_bonds))

    traced, n_bad, min_det = run_loop(
        pos, lambda p, t: (scores[t][0], scores[t][1], scores[t][2].reshape(n_poses, n_bonds)),
        table, noise_fn, bonds, mask)
    return eager.poses, traced, n_bad, min_det


def test_the_traced_loop_reproduces_the_graded_sampler_on_the_shipped_recipe():
    """The deliverable: same rollout as `sampling.sample`, with the shipped recipe."""
    cfg = S.SamplingConfig.from_recipe()
    eager, traced, n_bad, min_det = run_both(cfg)
    assert n_bad.item() == 0
    assert min_det is not None, "the determinant diagnostic comes back from the loop"
    assert rel_dev(traced, eager) < POSE_REL_TOL


@pytest.mark.parametrize("overrides", [
    {"ode": True, "temp_sampling": (1.0, 1.0, 1.0), "temp_psi": (0.0, 0.0, 0.0)},
    {"no_random": True},
    {"no_final_step_noise": False},
    {"temp_sampling": (1.7, 1.2, 0.8), "temp_psi": (0.3, 0.0, 0.5)},
    {"actual_steps": None},
])
def test_every_sampling_option_is_table_contents_not_a_branch(overrides):
    """Each control the CLI exposes must change only the table, not the loop body."""
    cfg = S.SamplingConfig.from_recipe(**overrides)
    eager, traced, _, _ = run_both(cfg)
    assert rel_dev(traced, eager) < POSE_REL_TOL


def test_in_float64_the_two_rollouts_are_the_same_function():
    """The real statement of equality: in float64 the gap is 2e-12 relative, i.e. nothing.

    Run in float32 the two differ by ~2e-05 relative after 19 steps, which is what rounding
    does to a chain of rotations with a tens-of-angstroms lever arm. Lifting the precision
    collapses the gap by seven orders of magnitude, which is what says the traced form is the
    same arithmetic rather than a close approximation of it.
    """
    torch.set_default_dtype(torch.float64)
    try:
        eager, traced, _, _ = run_both(S.SamplingConfig.from_recipe())
        assert rel_dev(traced, eager) < POSE_REL_TOL_F64
    finally:
        torch.set_default_dtype(torch.float32)


def test_ode_and_quiet_steps_zero_the_noise_column_rather_than_skipping_a_branch():
    cfg = S.SamplingConfig.from_recipe()
    table = coefficient_table(cfg, BOUNDS)
    assert table.noise[-1].abs().max().item() == 0.0, "the recipe sets no_final_step_noise"
    assert table.drawn[-1] is False and table.drawn[0] is True
    ode = coefficient_table(S.SamplingConfig.from_recipe(
        ode=True, temp_sampling=(1.0, 1.0, 1.0), temp_psi=(0.0, 0.0, 0.0)), BOUNDS)
    assert ode.noise.abs().max().item() == 0.0
    assert not any(ode.drawn)
    plain = coefficient_table(S.SamplingConfig.from_recipe(
        temp_sampling=(1.0, 1.0, 1.0), temp_psi=(0.0, 0.0, 0.0)), BOUNDS)
    assert (ode.score - 0.5 * plain.score).abs().max().item() < 1e-4


def test_the_guard_is_the_samplers_own_function():
    """One repair, both paths: this module imports sampling's (D-dfd-sampling-9)."""
    assert guard_scores is S.guard_scores


def test_the_guard_leaves_a_clean_score_alone():
    g = torch.Generator().manual_seed(1)
    score = torch.randn(10, 3, generator=g)
    clean, n_bad = guard_scores(score)
    assert torch.equal(clean, score) and n_bad.item() == 0
    assert torch.equal(S._repair_nans(score, score, score)[0], clean)


def test_the_guard_repairs_and_counts_the_poses():
    score = torch.tensor([[1.0, -2.0, 3.0], [float('nan'), 0.5, 0.5],
                          [2.0, 1.0, -1.0], [-4.0, 1.0, 1.0]])
    clean, n_bad = guard_scores(score)
    assert n_bad.item() == 1
    assert torch.isfinite(clean).all()
    assert torch.equal(S._repair_nans(score, score, score)[0], clean)


def test_an_infinity_is_repaired_and_does_not_poison_the_repair_value():
    """Upstream's `isnan` branch misses this; both of our paths repair it."""
    score = torch.tensor([[1.0, -2.0, 3.0], [float('inf'), 1.0, -1.0]])
    clean, n_bad = guard_scores(score)
    assert n_bad.item() == 1 and torch.isfinite(clean).all()
    assert torch.isfinite(S._repair_nans(score, score, score)[0]).all()
    # 1 % of the mean magnitude over the finite entries, the infinity left out of it.
    assert abs(clean[1, 0].item() - 0.01 * (1 + 2 + 3 + 1 + 1) / 5) < 1e-6
