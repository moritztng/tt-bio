"""Device-free regression tests for the pharma parity gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pharma_parity.py"


def _load():
    spec = importlib.util.spec_from_file_location("pharma_parity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_device_instability_does_not_hide_behind_widened_floor():
    mod = _load()

    verdict = mod.noise_floor_verdict(
        cross=[0.2, 0.3],
        ref_floor=[1.0, 1.1],
        dev_floor=[20.0, 22.0],
        metric="synthetic_distance",
    )

    assert verdict["within_noise_floor"], "small X should still retain its parity verdict"
    assert verdict["dev_over_ref_floor"] == pytest.approx(20.0)
    assert verdict["floor_inflated_by_dev"], (
        "a PASS made permissive by extreme device self-variance must carry an instability warning"
    )


def test_zero_reference_floor_skips_device_instability_ratio():
    mod = _load()

    verdict = mod.noise_floor_verdict(
        cross=[0.0],
        ref_floor=[0.0],
        dev_floor=[0.0],
        metric="deterministic_forward",
    )

    assert verdict["dev_over_ref_floor"] is None
    assert not verdict["floor_inflated_by_dev"]


def _sibling(name: str):
    path = SCRIPT.parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kabsch_rmsd_is_zero_for_a_rigidly_moved_copy():
    """Every RMSD the gate prints is meaningless if the superposition is wrong, and getting it
    wrong fails quietly: apply the inverse rotation instead of the rotation and two nearly
    aligned structures still score about right, while two in arbitrary relative orientation
    score roughly Rg apart. That is how a healthy 5-sample Protenix ensemble (0.9-2.6 A spread,
    every sample within 2.9 A of the reference) was reported as 2.26-20.73 A wide with a
    15.71 A rank-0 sample. A rigidly moved copy is the one input whose answer is known, so it
    is the one worth asserting.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    P = rng.normal(size=(64, 3)) * 10.0
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    Q = P @ R + np.array([5.0, -3.0, 2.0])

    impls = {
        "boltz2_fast_parity.kabsch": lambda a, b: _sibling("boltz2_fast_parity").kabsch(a, b)[0],
        "boltz2_affinity_parity._kabsch_rmsd": _sibling("boltz2_affinity_parity")._kabsch_rmsd,
    }
    for name, fn in impls.items():
        assert fn(P, Q) == pytest.approx(0.0, abs=1e-6), (
            f"{name} does not superpose: a rigid copy of a structure must score 0 A")
        assert fn(Q, P) == pytest.approx(0.0, abs=1e-6), f"{name} is not symmetric"
        # a real difference must still register
        assert 0.3 < fn(P, P + rng.normal(size=P.shape) * 0.5) < 0.9, f"{name} is not sensitive"


def _ens_run(width_max, per_sample):
    return {"n_samples": len(per_sample), "width": {"n": 1, "mean": width_max, "max": width_max},
            "per_sample_vs_ref": per_sample, "rank0_vs_ref": per_sample[0],
            "best_vs_ref": min(per_sample), "best_rank": per_sample.index(min(per_sample)),
            "selection_penalty": per_sample[0] - min(per_sample),
            "n_closer_than_rank0": sum(1 for v in per_sample[1:] if v < per_sample[0])}


def test_within_run_ensemble_bars_pass_a_healthy_run():
    mod = _load()
    ref = mod.summarize([1.4, 2.2, 2.9])          # the reference's own inter-seed floor
    v = mod.ensemble_verdict([_ens_run(2.6, [1.5, 2.0, 1.2, 1.2, 1.0])], ref)
    assert v["ensemble_width_ok"] and v["selection_ok"]


def test_within_run_ensemble_bars_fail_a_wide_ensemble_and_an_inverted_ranker():
    """The defect class no instrument could fail before: five samples one predict call returned
    that disagree by 20 A, with the model shipping the worst of them. Both bars are scaled off
    the reference's own inter-seed floor, so this is not a constant that a hard target trips."""
    mod = _load()
    ref = mod.summarize([1.4, 2.2, 2.9])
    v = mod.ensemble_verdict([_ens_run(20.7, [15.7, 2.3, 10.1, 20.4, 6.1])], ref)
    assert not v["ensemble_width_ok"], "a 20 A ensemble must fail the width bar"
    assert not v["selection_ok"], "shipping the 15.7 A sample over the 2.3 A one must fail"
    assert v["rank0_was_best"] == 0


def test_within_run_ensemble_asserts_nothing_without_a_reference_floor():
    """A single-seed fixture gives no R, so there is no bar. Report the numbers, claim nothing."""
    mod = _load()
    v = mod.ensemble_verdict([_ens_run(20.7, [15.7, 2.3])], mod.summarize([]))
    assert "ensemble_width_ok" not in v and "selection_ok" not in v
    assert v["width_max"] == 20.7


def test_single_sample_run_has_no_ensemble_verdict():
    """Boltz-2 and the OpenDDE structure legs fold one sample; there is nothing to spread."""
    mod = _load()
    assert mod.ensemble_verdict([None], mod.summarize([1.4, 2.2])) == {"n_runs": 0}
