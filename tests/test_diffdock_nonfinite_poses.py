"""A pose that goes non-finite is a result, not a crash.

Three of `tt-bio dock`'s output tests died in `modify_conformer`'s Kabsch SVD ("input matrix
contained non-finite values") under torch 2.8.0 and passed under 2.13. The torch version only
changes which starting pose is drawn, so it was luck of the draw, not an environment
artifact: `sampling._repair_nans` keyed on `isnan` and let a head that went to +-inf through
unrepaired. The guard now keys on `isfinite` (one function, shared with the device driver),
and the rollout freezes a pose it cannot advance instead of losing the complex with it.

These tests drive the infinity directly rather than waiting for a draw to produce one.
"""

from __future__ import annotations

import pytest
import torch

from tt_bio.diffdock import sampling

BOUNDS = sampling.SigmaBounds(tr_sigma_min=0.1, tr_sigma_max=19.0, rot_sigma_min=0.03,
                              rot_sigma_max=1.55, tor_sigma_min=0.0314, tor_sigma_max=3.14)


def config(**kw):
    return sampling.SamplingConfig.from_recipe(inference_steps=6, actual_steps=5, **kw)


def ligand():
    """Four atoms, one rotatable bond: enough to reach the torsion path and the SVD."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.2, 1.2, 0.0], [3.6, 1.1, 0.4]])
    edge_index = torch.tensor([[1, 2]])
    mask_rotate = torch.tensor([[False, False, False, True]])
    return pos, edge_index, mask_rotate


def rollout(bad_rows, *, value=float("inf"), batch=3, head=0, steps_bad=(2,)):
    """Roll `batch` poses out with *head* set to *value* on *bad_rows* at *steps_bad*."""
    pos, edge_index, mask_rotate = ligand()
    lig_pos = pos.unsqueeze(0).repeat(batch, 1, 1)

    def score_fn(step):
        heads = [-0.05 * step.lig_pos.mean(dim=1) / (step.tr_sigma ** 2 + 1.0),
                 torch.zeros(batch, 3),
                 torch.zeros(batch * mask_rotate.shape[0])]
        if step.step in steps_bad:
            h = heads[head].reshape(batch, -1).clone()
            h[list(bad_rows)] = value
            heads[head] = h.reshape(heads[head].shape)
        return tuple(heads)

    return sampling.sample(score_fn, lig_pos, cfg=config(), bounds=BOUNDS,
                           noise=sampling.Noise(seed=11), edge_index=edge_index,
                           mask_rotate=mask_rotate, trajectory=True)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_score_head_is_repaired_not_fatal(value):
    """The crash itself: +-inf reached the batched SVD and took every pose down."""
    out = rollout([1], value=value)
    assert torch.isfinite(out.poses).all()
    # The repair nudges the pose; it does not fail it, so nothing is excluded from ranking.
    assert not bool(out.failed.any())


def test_a_pose_that_goes_non_finite_is_frozen_and_recorded():
    """No repair can scale an overflowing update: freeze that pose, keep the others."""
    out = rollout([1], value=1e38, steps_bad=(2,))
    assert out.failed.tolist() == [False, True, False]
    assert torch.isfinite(out.poses).all()
    # The step it could not take left it exactly where it was; later steps still run, and
    # the mark stays on it either way, so the ranking never has to trust the recovery.
    assert torch.equal(out.trajectory[3][1], out.trajectory[2][1])


def test_one_bad_pose_does_not_move_the_others():
    """The failure is row-local. The other poses are what a clean run gives, value for value."""
    bad = rollout([1], value=1e38)
    clean = rollout([], value=1e38)
    for row in (0, 2):
        assert torch.equal(bad.poses[row], clean.poses[row])


def test_all_poses_failing_is_still_a_result():
    out = rollout([0, 1, 2], value=1e38)
    assert bool(out.failed.all())
    assert torch.isfinite(out.poses).all()


def test_a_non_finite_starting_pose_is_failed_at_the_door():
    pos, edge_index, mask_rotate = ligand()
    lig_pos = pos.unsqueeze(0).repeat(2, 1, 1)
    lig_pos[0, 0, 0] = float("nan")

    def score_fn(step):
        return (torch.zeros(2, 3), torch.zeros(2, 3),
                torch.zeros(2 * mask_rotate.shape[0]))

    out = sampling.sample(score_fn, lig_pos, cfg=config(), bounds=BOUNDS,
                          noise=sampling.Noise(seed=3), edge_index=edge_index,
                          mask_rotate=mask_rotate)
    assert out.failed.tolist() == [True, False]
    assert torch.isfinite(out.poses).all()


def test_a_clean_rollout_is_untouched_by_the_guard():
    """The guard must change nothing on a run that has numbers. Bit-identical, not close."""
    a = rollout([], steps_bad=())
    b = rollout([], steps_bad=())
    assert torch.equal(a.poses, b.poses)
    assert not bool(a.failed.any())


def test_a_rigid_ligand_rolls_out():
    """A ligand with no rotatable bond has an empty torsion head, and the guard reshaped it.

    `7OP9_06K` (dichloropurine, 0 rotatable bonds) died at step 0 of `tt-bio dock` with
    "cannot reshape tensor of 0 elements into shape [0, -1]". Nothing downstream touches the
    head -- `torsion` is already False when `mask_rotate` is empty -- so the guard was the
    only thing standing between a rigid ligand and a docked pose.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.1, 1.2, 0.0]])
    lig_pos = pos.unsqueeze(0).repeat(2, 1, 1)
    edge_index = torch.zeros(0, 2, dtype=torch.long)
    mask_rotate = torch.zeros(0, 3, dtype=torch.bool)

    def score_fn(step):
        return (torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(0))

    out = sampling.sample(score_fn, lig_pos, cfg=config(), bounds=BOUNDS,
                          noise=sampling.Noise(seed=5), edge_index=edge_index,
                          mask_rotate=mask_rotate)
    assert torch.isfinite(out.poses).all()
    assert not bool(out.failed.any())


def test_the_guard_counts_nothing_bad_in_an_empty_head():
    score = torch.zeros(0)
    repaired, n_bad = sampling.guard_scores(score)
    assert repaired.shape == score.shape and int(n_bad) == 0
