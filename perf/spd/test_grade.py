"""grade.no_op: a test arm that matches base bit for bit must show its lever ran (ge 11-19 read PASS as a no-op)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
grade = pytest.importorskip("grade")


def rep(arm, seed, digest, levers="", latches=None):
    return dict(arch="bh", arm=arm, input="7XYZ", seed=seed, err=None, digest=digest,
                arm_env={"TT_BIO_LEVERS": levers} if levers else {}, latches=latches or {})


def test_identical_without_served_latch_is_no_op():
    reps = [rep("b", s, f"d{s}") for s in range(4)] + [rep("e", s, f"d{s}", "edm_device") for s in range(4)]
    assert "bit for bit on all 4" in grade.no_op(reps, "bh", "b", "e")


def test_identical_with_served_test_only_latch_passes():
    lat = {"trimul_t1_lnpass": {"served": 3}}
    reps = [rep("b", s, f"d{s}") for s in range(4)] + [rep("e", s, f"d{s}", "trimul_t1_lnpass", lat) for s in range(4)]
    assert grade.no_op(reps, "bh", "b", "e") is None


def test_any_differing_digest_is_not_no_op():
    reps = [rep("b", s, f"d{s}") for s in range(4)] + [rep("e", s, f"d{s}" if s else "x", "edm_device") for s in range(4)]
    assert grade.no_op(reps, "bh", "b", "e") is None


def test_no_digests_is_not_no_op():
    reps = [rep("b", s, None) for s in range(4)] + [rep("e", s, None, "edm_device") for s in range(4)]
    assert grade.no_op(reps, "bh", "b", "e") is None
