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
