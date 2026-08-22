"""Card-free tests for the tap gate's envelope adjudication.

The expensive proof is adversarial and lives in the pass log: three transcription bugs injected
into the monomer run, two of which move the model (`pre_compose` rotating by the new frame, the
sidechain reading its two projections in the wrong order) and are caught at PCC 0.65 against an
envelope of 0.99. What is testable without the 373 MB checkpoint is the decision rule itself,
which is the part that decides whether a real defect gets excused.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "af2_port"))

import tap_gate  # noqa: E402


def _row(pcc=1.0, d_mean=0.0, d_std=0.0, d_sumsq=0.0):
    return {"pcc": pcc, "d_mean": d_mean, "d_std": d_std, "d_sumsq": d_sumsq}


def test_a_miss_inside_the_envelope_is_adjudicated_in():
    assert tap_gate._in_envelope(_row(pcc=0.97), _row(pcc=0.95))


def test_a_miss_with_a_collapsed_envelope_is_a_defect():
    """A wrong module is wrong in both dtypes, so the two arms agree and the envelope closes."""
    assert not tap_gate._in_envelope(_row(pcc=0.65), _row(pcc=0.999))


def test_only_the_measures_that_broke_their_bar_are_adjudicated():
    """A tap that misses on correlation need not justify statistics that already pass."""
    row = _row(pcc=0.97, d_mean=0.001)
    assert tap_gate._in_envelope(row, _row(pcc=0.95, d_mean=0.0))


def test_a_statistics_miss_is_adjudicated_on_the_statistics():
    assert tap_gate._in_envelope(_row(d_std=0.05), _row(d_std=0.06))
    assert not tap_gate._in_envelope(_row(d_std=0.05), _row(d_std=0.01))


def test_nothing_to_adjudicate_is_not_a_pass():
    """`_in_envelope` is only ever asked about a failing row; a clean one must not slip through."""
    assert not tap_gate._in_envelope(_row(), _row())


def test_an_arm_scored_against_itself_is_exact():
    """`as_reference` has to reproduce the reference's own format, subsample positions included."""
    values = torch.arange(60.0).reshape(3, 20)
    ref = {"t/shape": np.array([3, 20]), "t/idx": np.array([0, 7, 19, 41]),
           "t/val": np.zeros(4), "t/stats": np.zeros(5)}
    scored = tap_gate.score_one(tap_gate.as_reference(ref, "t", values), "t", values)
    assert scored["verdict"] == "PASS"
    assert scored["pcc"] > 1 - 1e-12
    assert scored["sampled"] == 4
