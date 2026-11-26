"""Horn's quaternion against the SVD Kabsch the graded sampler uses.

`tt_bio.diffdock.sampling.kabsch_batch` is upstream's arithmetic: an SVD per pose and a `where`
on the determinant for the reflection case. `tt_bio.diffdock.kabsch.kabsch_horn` has to agree
with it on both cases while issuing one fixed-shape program -- that is the whole reason it
exists, so both cases are graded here, and the reflection one is constructed deliberately
rather than hoped for.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock.kabsch import SQUARINGS, kabsch_horn, proper_rotation  # noqa: E402
from tt_bio.diffdock.sampling import kabsch_batch, torsion_update_batch  # noqa: E402

# float32 floor measured below; float64 shows the method itself is exact to 1e-14.
FLOOR_F32 = 2e-6


def clouds(n_atoms=24, n_poses=10, seed=11, torsion=0.3):
    """A conformer and its torsion-perturbed copy -- the pair the in-loop Kabsch aligns."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.cumsum(torch.randn(n_poses, n_atoms, 3, generator=g), dim=1) * 1.5
    bonds = torch.tensor([(k, k + 1) for k in range(n_atoms - 1)])
    mask = torch.zeros(len(bonds), n_atoms, dtype=torch.bool)
    for k in range(len(bonds)):
        mask[k, k + 1:] = True
    updates = torch.randn(n_poses, len(bonds), generator=g) * torsion
    return torsion_update_batch(pos, bonds, mask, updates), pos


def test_it_returns_a_rotation_not_merely_an_orthogonal_matrix():
    a, b = clouds()
    r, _, _ = kabsch_horn(a, b)
    eye = torch.eye(3).expand_as(r)
    assert (r @ r.transpose(1, 2) - eye).abs().max().item() < 1e-5
    assert (torch.linalg.det(r) - 1).abs().max().item() < 1e-5


def test_it_matches_samplings_svd_on_a_torsion_perturbed_conformer():
    a, b = clouds(torsion=0.08)
    r, t, det = kabsch_horn(a, b)
    want_r, want_t = kabsch_batch(a, b)
    assert (det > 0).all(), "this fixture is the reflection-free case"
    assert (r - want_r).abs().max().item() < FLOOR_F32
    assert (t - want_t).abs().max().item() < 1e-4


def test_it_matches_samplings_svd_in_the_reflection_case_too():
    """The case the polar factor cannot reach, which is why this module uses Horn's.

    Mirroring one cloud forces `det(H) < 0`, where sampling (and upstream) swap in
    `V diag(1,1,-1) U^T`. Horn's quaternion has no branch to take and lands on the same
    rotation; the orthogonal polar factor would return the improper matrix and be ~1.9 out.
    """
    a, b = clouds()
    a = a.clone()
    a[..., 2] *= -1
    r, _, det = kabsch_horn(a, b)
    want_r, _ = kabsch_batch(a, b)
    assert (det < 0).all(), "the fixture must actually be the reflecting case"
    assert (torch.linalg.det(r) - 1).abs().max().item() < 1e-4, "still a proper rotation"
    assert (r - want_r).abs().max().item() < FLOOR_F32


@pytest.mark.parametrize("mirror", [False, True])
def test_ten_squarings_reach_the_float32_floor(mirror):
    """Why SQUARINGS is 10: six is visibly unconverged, ten is at the floor and eleven is not
    better. A fixed count, not a tuned one -- a trace cannot iterate to a tolerance."""
    a, b = clouds()
    if mirror:
        a = a.clone()
        a[..., 2] *= -1
    want, _ = kabsch_batch(a.double(), b.double())
    h = (a - a.mean(1, keepdim=True)).transpose(1, 2) @ (b - b.mean(1, keepdim=True))
    errs = {n: (proper_rotation(h, squarings=n).double() - want).abs().max().item()
            for n in (6, 8, SQUARINGS)}
    assert errs[6] > errs[8] >= errs[SQUARINGS], errs
    assert errs[SQUARINGS] < FLOOR_F32
    # 8 already reaches the floor on the easier of the two fixtures; 10 is the margin that
    # keeps the mirrored one there too, and costs two more 4x4 matmuls.


def test_the_method_itself_is_exact_in_float64():
    """Separating the method from its precision: in float64 it agrees with the SVD to 1e-13."""
    a, b = clouds()
    r, _, _ = kabsch_horn(a.double(), b.double())
    want, _ = kabsch_batch(a.double(), b.double())
    assert (r - want).abs().max().item() < 1e-12
