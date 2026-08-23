"""The fourth `af2_easy` criterion, and the three ways a coordinate metric goes quietly wrong.

`bound_unbound_rmsd.py` has a sign convention, a reflection branch and a rounding rule, and none of
the three announces itself when it is wrong: a mirrored superposition, an off-by-one in the binder
slice and an unrounded threshold all return a plausible number. So each is pinned here, against
values computable by hand rather than against a recorded run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "af2_port"))

import bound_unbound_rmsd as bu  # noqa: E402


def _cloud(n=40, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=8.0, size=(n, 3))


def _rotation(seed=11):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_identical_clouds_are_zero():
    p = _cloud()
    assert bu.kabsch_rmsd(p, p.copy()) == pytest.approx(0.0, abs=1e-12)


def test_rigid_motion_is_removed():
    """A rotation plus a translation is exactly what the superposition exists to absorb."""
    p = _cloud()
    q = p @ _rotation() + np.array([13.0, -4.0, 71.0])
    assert bu.kabsch_rmsd(p, q) < 1e-9


def test_uniform_displacement_gives_its_own_magnitude():
    """Every atom pushed off its partner by the same vector, alternating so no rigid motion can
    absorb it: the RMSD is that vector's length."""
    p = _cloud(n=41)
    step = np.array([0.3, -0.4, 1.2])          # |step| = 1.3
    q = p + step * ((-1.0) ** np.arange(len(p)))[:, None]
    assert bu.kabsch_rmsd(p, q) == pytest.approx(np.linalg.norm(step), rel=2e-2)


def test_reflection_branch_fires():
    """Without the det < 0 fix, SVD hands back an improper rotation and a mirrored cloud scores
    ~0 -- a mirror image of a binder is a different fold, so it must not."""
    p = _cloud()
    mirrored = p * np.array([1.0, 1.0, -1.0])
    assert bu.kabsch_rmsd(p, mirrored) > 1.0
    u, _, vt = np.linalg.svd((p - p.mean(0)).T @ (mirrored - mirrored.mean(0)))
    assert np.linalg.det(vt.T @ u.T) < 0, "this fixture no longer exercises the reflection branch"


def test_length_mismatch_raises():
    """Upstream's Binder_ variant returns None rather than pairing residues up; a silent number
    from a mismatched slice would be the off-by-one nobody sees."""
    with pytest.raises(ValueError):
        bu.kabsch_rmsd(_cloud(n=40), _cloud(n=39))


def test_non_ca_shape_raises():
    with pytest.raises(ValueError):
        bu.kabsch_rmsd(np.zeros((10, 3, 3)), np.zeros((10, 3, 3)))


@pytest.mark.parametrize("value,accepted", [(3.494, True), (3.4949, True), (3.495, False),
                                            (3.5, False), (3.504, False), (0.2658, True)])
def test_threshold_rounds_to_two_decimals_like_upstream(value, accepted):
    """main_af2_monomer.py:164 rounds before comparing, so 3.495 rejects and 3.494 accepts."""
    assert bu.accepts(value) is accepted


def test_missing_rmsd_never_accepts():
    assert bu.accepts(None) is False


def test_four_criterion_verdict_is_the_conjunction():
    """The fourth criterion can only remove accepts, which is the argument that bounds the host
    coverage this measurement needs."""
    import filter_flip_rate as ffr
    scalars = {"plddt": 0.91, "i_ptm": 0.62, "i_pae": 0.28}
    assert ffr.accept3(scalars) is True
    assert ffr.accept4(scalars, 0.27) is True
    assert ffr.accept4(scalars, 9.9) is False
    rejected = {"plddt": 0.91, "i_ptm": 0.41, "i_pae": 0.28}
    assert ffr.accept3(rejected) is False
    assert ffr.accept4(rejected, 0.27) is False


def test_coverage_assertion_fires_on_an_accepted_design_without_an_rmsd():
    """A design accepted on either arm under three criteria and missing an RMSD makes the
    four-criterion number unstated rather than equal to the three-criterion one."""
    import filter_flip_rate as ffr
    accepted = {"plddt": 0.91, "i_ptm": 0.62, "i_pae": 0.28}
    rejected = {"plddt": 0.5, "i_ptm": 0.1, "i_pae": 0.9}
    yes, no = {"a": {"ref": accepted}}, {"a": {"ref": rejected}}
    with pytest.raises(AssertionError):
        ffr.check_rmsd_coverage(yes, yes, {}, {})
    # accepted on one arm only is still in the union
    with pytest.raises(AssertionError):
        ffr.check_rmsd_coverage(no, yes, {"a": 0.3}, {})
    # reject-on-both needs no RMSD: the conjunction cannot move it
    assert ffr.check_rmsd_coverage(no, no, {}, {})["n_union"] == 0
    assert ffr.check_rmsd_coverage(yes, yes, {"a": 0.3}, {"a": 0.4})["n_union"] == 1


def test_upstream_pdb_roundtrip_preserves_the_cloud():
    """The cross-check goes through a PDB, so its writer has to be lossless to three decimals or
    the tolerance is measuring the writer."""
    p = np.round(_cloud(n=12), 3)
    path = Path("/tmp/af2ig_rmsd_test/roundtrip.pdb")
    path.parent.mkdir(parents=True, exist_ok=True)
    bu.write_ca_pdb(path, [("A", p), ("B", p + 5.0)])
    got = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    for line in path.read_text().splitlines() if line.startswith("ATOM")])
    assert got.shape == (24, 3)
    assert np.abs(got[:12] - p).max() < 5e-4
