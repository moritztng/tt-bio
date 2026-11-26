"""The SO(3) and torus score-norm tables the output heads scale by.

Two different kinds of grade, because the two tables are different kinds of object:

* **SO(3) is deterministic.** It is a truncated series with a closed definition, so the bar
  is upstream's own value, tight: 2.7e-11 max relative over the eps range the model reaches.
* **The torus table is Monte Carlo with no seed** (``utils/torus.py`` draws 10 000 samples at
  import time). It therefore has no fixed value to match. Measured over three upstream
  processes on 2026-11-26: 0.8 % median and 3.6 % max relative spread, per grid point. The
  bar is that our deterministic quadrature sits inside that spread, and grading it tighter
  would be grading against one draw.

The upstream comparisons are skipped when the reference clone is absent.
"""

import math
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from tt_bio.diffdock import schedules as S

UPSTREAM = Path("/home/moritz/scratch/dfd/upstream")
# The sigma ranges the released DiffDock-L actually samples, from model_parameters.yml.
ROT_SIGMA_MIN, ROT_SIGMA_MAX = 0.03, 1.55
TOR_SIGMA_MIN, TOR_SIGMA_MAX = 0.0314, 3.14
TORUS_MC_SPREAD = 0.036      # max run-to-run relative spread over three upstream processes


def _so3_index(eps):
    return int(round((math.log10(eps) - math.log10(S.SO3_MIN_EPS))
                     / (math.log10(S.SO3_MAX_EPS) - math.log10(S.SO3_MIN_EPS)) * S.SO3_N_EPS))


def test_tables_have_upstreams_shapes():
    assert S.so3_score_norm_table().shape == (S.SO3_N_EPS,)
    assert S.torus_score_norm_table().shape == (S.TORUS_SIGMA_N + 1,)


def test_so3_is_finite_and_smooth_where_the_model_reaches():
    """No spikes in [0.03, 1.55].

    This is the test that catches the matmul trap: reassociating the series into a matrix
    product leaves the curve finite and plausible on average but grows isolated spikes up to
    1e130 at points where ``f(w)`` has cancelled to ~1e-14. A second-difference bound sees
    that immediately; a mean relative error does not.
    """
    t = S.so3_score_norm_table()
    lo, hi = _so3_index(ROT_SIGMA_MIN), _so3_index(ROT_SIGMA_MAX)
    seg = t[lo:hi + 1]
    assert np.isfinite(seg).all()
    assert (seg > 0).all()
    curv = np.abs(seg[1:-1] - 0.5 * (seg[:-2] + seg[2:])) / np.abs(seg[1:-1])
    assert curv.max() < 1e-3, f"second difference {curv.max():.3e}: the curve has spikes"
    # it is a decreasing function of the noise level
    assert seg[0] > seg[-1] * 10


def test_torus_is_finite_and_decreasing_where_the_model_reaches():
    t = S.torus_score_norm_table()
    assert np.isfinite(t).all() and (t > 0).all()
    s = np.array([TOR_SIGMA_MIN, 0.3, 1.0, TOR_SIGMA_MAX])
    vals = S.torus_score_norm(s)
    assert (np.diff(vals) < 0).all(), vals


def test_the_lookup_is_a_step_function_not_an_interpolation():
    """Upstream rounds to a grid index, so nearby sigmas can share or jump a value.

    Smoothing this would be a silent change to the model's output scaling, and would show up
    only as a small bias in the rot and tor heads.
    """
    grid = S.so3_eps_grid()
    a = S.so3_score_norm(torch.tensor([grid[1000]]))
    b = S.so3_score_norm(torch.tensor([grid[1000] * 1.0001]))
    assert torch.equal(a, b)
    c = S.so3_score_norm(torch.tensor([grid[1001]]))
    assert not torch.equal(a, c)


def test_so3_clamps_outside_its_grid():
    far = S.so3_score_norm(torch.tensor([1e-9, 1e9]))
    assert torch.isfinite(far).all()


@pytest.mark.skipif(not (UPSTREAM / ".so3_exp_score_norms4.npy").exists(),
                    reason="upstream clone with its .npy caches is not present")
def test_so3_matches_upstream_where_the_model_reaches():
    up = np.load(UPSTREAM / ".so3_exp_score_norms4.npy")
    ours = S.so3_score_norm_table()
    lo, hi = _so3_index(ROT_SIGMA_MIN), _so3_index(ROT_SIGMA_MAX)
    sl = slice(lo, hi + 1)
    assert np.isfinite(up[sl]).all() and np.isfinite(ours[sl]).all()
    rel = np.abs(ours[sl] - up[sl]) / np.abs(up[sl])
    assert rel.max() < 1e-9, f"max relative {rel.max():.3e}"


@pytest.mark.skipif(not (UPSTREAM / ".so3_exp_score_norms4.npy").exists(),
                    reason="upstream clone with its .npy caches is not present")
def test_the_non_finite_so3_entries_are_upstreams_too_and_are_unreachable():
    """Both tables go non-finite at the same 19 points, all below the model's sigma floor.

    Worth pinning: if ours ever goes non-finite somewhere upstream does not, the series has
    been changed, and the first symptom in a real run would be a NaN score at one noise level
    out of 2000 -- a failure that looks random.
    """
    up = np.load(UPSTREAM / ".so3_exp_score_norms4.npy")
    ours = S.so3_score_norm_table()
    bad_up = set(np.where(~np.isfinite(up))[0].tolist())
    bad_ours = set(np.where(~np.isfinite(ours))[0].tolist())
    assert bad_ours == bad_up, f"ours-only {sorted(bad_ours - bad_up)[:5]}"
    assert max(bad_up) < _so3_index(ROT_SIGMA_MIN)


@pytest.mark.skipif(not os.environ.get("DFD_SLOW_UPSTREAM"),
                    reason="importing upstream's utils/torus.py recomputes its 10000-draw "
                           "Monte Carlo and loads two 200 MB caches: over 23 min. Set "
                           "DFD_SLOW_UPSTREAM=1 to run it.")
@pytest.mark.skipif(not (UPSTREAM / ".score.npy").exists(),
                    reason="upstream clone with its .npy caches is not present")
def test_torus_sits_inside_upstreams_monte_carlo_spread():
    import sys
    sys.path.insert(0, str(UPSTREAM))
    pytest.importorskip("tqdm")
    from utils import torus           # recomputes its own unseeded sample on import
    ours = S.torus_score_norm_table()
    theirs = torus.score_norm_
    s = S.torus_sigma_grid()
    live = (s >= TOR_SIGMA_MIN) & (s <= TOR_SIGMA_MAX)
    rel = np.abs(ours[live] - theirs[live]) / np.abs(theirs[live])
    assert np.median(rel) < TORUS_MC_SPREAD, f"median {np.median(rel):.3e}"
    assert np.percentile(rel, 99) < 3 * TORUS_MC_SPREAD, f"p99 {np.percentile(rel, 99):.3e}"


def test_narrow_torus_matches_the_unwrapped_closed_form():
    """An independent grade that needs no upstream: for a narrow wrapped normal the images
    vanish, the score is ``-x / sigma^2`` and its mean square is exactly ``1 / sigma^2``.
    Measured 7e-14 relative at the narrowest grid point."""
    s = S.torus_sigma_grid()
    t = S.torus_score_norm_table()
    narrow = s < 0.05
    rel = np.abs(t[narrow] * s[narrow] ** 2 - 1.0)
    assert rel.max() < 1e-10, f"max relative {rel.max():.3e}"
