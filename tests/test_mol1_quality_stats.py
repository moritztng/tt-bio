"""The MOL1 quality statistics pass a re-seed of the same model and fail a real accuracy drop (CPU only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf" / "mol1_quality"))
import stats  # noqa: E402


def synth(shift=0.0, n=40, seeds=5, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    truth = rng.uniform(0.1, 0.9, n)  # per-target difficulty, shared by every arm
    rows = []
    for arm, k in (("gpu", 0), ("gpu_self", 1), ("tt", 2)):
        for c in range(n):
            for s in range(seeds):
                q = float(np.clip(truth[c] + rng.normal(0, 0.08) - (shift if arm == "tt" else 0), 0, 1))
                rows.append({"arm": arm, "case": f"c{c}", "seed": s, "dockq": q, "lddt": q,
                             "iptm": float(np.clip(q + rng.normal(0, 0.05), 0, 1)),
                             "plddt": float(np.clip(q + rng.normal(0, 0.05), 0, 1)),
                             "ranking_score": float(rng.uniform())})
    return rows


def test_reseed_passes():
    assert stats.verdict(synth(0.0), test="tt")["verdict"] == "PASS"


def test_accuracy_drop_fails():
    r = stats.verdict(synth(0.15), test="tt")
    assert r["verdict"] == "FAIL"
    assert r["metrics"]["dockq"]["pass"] is False


def test_too_few_targets_is_insufficient():
    assert stats.verdict(synth(0.0, n=3), test="tt")["verdict"] == "INSUFFICIENT"
