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


# ---------------------------------------------------------------------------
# the overflowed Kabsch covariance: every number finite, and the SVD still raises
# (D-dfd-sampling-10)
# ---------------------------------------------------------------------------

def blown_up_batch(mag=1e30, row=3, batch=10):
    """Ten poses, one of them carried out to *mag* with its geometry along.

    This is where the four eval complexes end up (7AVI_S2Q, 7TBU_S3P, 7X5N_5M5, 7ZCC_OGA):
    the tr/rot heads reach ~1e16 -- upstream's own network returns the same magnitude on
    these poses, PCC 0.999999997, `state/dfd/BLOWUP-2026-11-27.md` -- and a few steps later
    the sample's coordinates are far enough out that the *products* in the Kabsch
    correlation matrix overflow fp32 while every coordinate is still finite. No guard that
    keys on a non-finite score or a non-finite update sees anything wrong, and
    `torch.linalg.svd` raises for all ten samples at once.
    """
    pos, edge_index, mask_rotate = ligand()
    lig_pos = pos.unsqueeze(0).repeat(batch, 1, 1)
    lig_pos[row] = (lig_pos[row] + 1.0) * mag
    assert torch.isfinite(lig_pos).all()
    return lig_pos, edge_index, mask_rotate


def blown_up_rollout(mag=1e30, row=3, batch=10):
    lig_pos, edge_index, mask_rotate = blown_up_batch(mag, row, batch)

    def score_fn(step):
        return (-0.05 * step.lig_pos.mean(dim=1) / (step.tr_sigma ** 2 + 1.0),
                torch.zeros(batch, 3),
                torch.zeros(batch * mask_rotate.shape[0]))

    return sampling.sample(score_fn, lig_pos, cfg=config(), bounds=BOUNDS,
                           noise=sampling.Noise(seed=11), edge_index=edge_index,
                           mask_rotate=mask_rotate, trajectory=True)


def test_an_overflowing_sample_does_not_take_the_batched_svd_down():
    """The campaign's last failing eval condition: the complex writes nine poses, not none."""
    out = blown_up_rollout()
    assert out.failed.tolist() == [i == 3 for i in range(10)]
    assert torch.isfinite(out.poses).all()
    # frozen at the last coordinates it actually had, nothing invented for it
    assert torch.equal(out.poses[3], out.trajectory[0][3])


def test_the_nine_survivors_are_bit_identical_to_a_run_without_the_bad_sample():
    """Row-local, to the bit: an overflowing sample changes no other sample's arithmetic."""
    bad = blown_up_rollout()
    clean = blown_up_rollout(mag=1.0)
    for row in range(10):
        if row == 3:
            continue
        assert torch.equal(bad.poses[row], clean.poses[row]), row
    assert not bool(clean.failed.any())


def test_the_overflow_is_caught_before_the_svd_is_called(monkeypatch):
    """Detection is on the SVD's input, so `linalg.svd` is never handed a non-finite matrix."""
    seen = []
    real_svd = torch.linalg.svd

    def spy(x, *a, **kw):
        seen.append(bool(torch.isfinite(x).all()))
        return real_svd(x, *a, **kw)

    monkeypatch.setattr(torch.linalg, "svd", spy)
    out = blown_up_rollout()
    assert seen, "the fixture must actually reach the SVD"
    assert all(seen), "a non-finite matrix reached the SVD"
    assert bool(out.failed[3])


def test_both_kabsch_routes_flag_the_same_overflowed_sample():
    """One check, both paths: the host SVD route and the device Horn route agree."""
    from tt_bio.diffdock.kabsch import kabsch_horn

    a = torch.randn(4, 12, 3)
    b = a + 0.01
    a[2] = a[2] * 1e30
    b[2] = b[2] * 1e30
    _, _, host_bad = sampling.kabsch_batch(a, b)
    _, _, _, device_bad = kabsch_horn(a, b)
    assert host_bad.tolist() == [False, False, True, False]
    assert device_bad.tolist() == host_bad.tolist()


def test_a_complex_whose_every_sample_overflows_is_one_failed_result():
    """All ten gone is one `failures.csv` row, not a traceback."""
    pos, edge_index, mask_rotate = ligand()
    lig_pos = ((pos + 1.0) * 1e30).unsqueeze(0).repeat(3, 1, 1)

    def score_fn(step):
        return (torch.zeros(3, 3), torch.zeros(3, 3), torch.zeros(3 * mask_rotate.shape[0]))

    out = sampling.sample(score_fn, lig_pos, cfg=config(), bounds=BOUNDS,
                          noise=sampling.Noise(seed=5), edge_index=edge_index,
                          mask_rotate=mask_rotate)
    assert bool(out.failed.all())
    assert torch.isfinite(out.poses).all()
