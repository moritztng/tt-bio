"""Harness shim for running upstream OpenFold3 `test_training_full.py` off CUDA.

Loaded with `-p of3t_theirtest_shim`. The upstream test file, the upstream model code and
the upstream runner yaml all stay byte-identical; every adaptation lives here.

SHIM 1 -- drop the `@skip_unless_cuda_available()` mark (test_training_full.py:174).
  The mark resolves to `pytest.mark.skipif(True, reason="Requires cuda; found cpu")` via
  compare_utils.py:155. Removing it is what lets the body run at all. It does not touch an
  assertion, a tolerance, a step count or the data.

SHIM 2 -- set `pl_trainer_args.accelerator` in the dict the test loads from the runner yaml.
  `build_runner_yaml_config` (scripts/datasets/pdb_subset_helpers.py:548) emits no
  `accelerator` key, so pydantic fills the default "gpu" (entry_points/validator.py:126) and
  Lightning refuses to start. The value comes from $OF3T_ACCELERATOR; unset means no change.
  This is the accelerator shim / device argument the brief names as legitimate.

SHIM 3 -- DIAGNOSTIC, off unless $OF3T_EVAL_TRITON_OFF=1. Clears
  `memory.eval.use_triton_triangle_kernels`. This one changes the MODEL CONFIGURATION, not
  the device, so a run with it on is NOT a pass of upstream's test.

Nothing else is patched: no assertion, no tolerance, no timeout, no max_epochs, no
epoch_len, no dataset, no seed.
"""

import os

import pytest

_CUDA_SKIP_REASON = "Requires cuda"
_CASE_IDS = ("smoke", "full_subset")


def pytest_collection_modifyitems(config, items):
    """SHIM 1."""
    for item in items:
        if not item.nodeid.endswith(tuple(f"test_train[{c}]" for c in _CASE_IDS)):
            continue
        kept = []
        for mark in item.own_markers:
            if mark.name == "skipif" and _CUDA_SKIP_REASON in str(
                mark.kwargs.get("reason", "")
            ):
                continue
            kept.append(mark)
        item.own_markers = kept


def _patch_loader(monkeypatch, mutate):
    from openfold3.core.config import config_utils

    original = config_utils.load_yaml

    def patched(path, *a, **kw):
        cfg = original(path, *a, **kw)
        if isinstance(cfg, dict):
            mutate(cfg)
        return cfg

    monkeypatch.setattr(config_utils, "load_yaml", patched)


@pytest.fixture(autouse=True)
def _of3t_accelerator(monkeypatch):
    """SHIM 2."""
    accel = os.environ.get("OF3T_ACCELERATOR")
    if not accel:
        return

    def mutate(cfg):
        cfg.setdefault("pl_trainer_args", {})["accelerator"] = accel

    _patch_loader(monkeypatch, mutate)


@pytest.fixture(autouse=True)
def _of3t_eval_triton(monkeypatch):
    """SHIM 3."""
    if os.environ.get("OF3T_EVAL_TRITON_OFF") != "1":
        return

    def mutate(cfg):
        mem = (
            cfg.setdefault("model_update", {})
            .setdefault("custom", {})
            .setdefault("settings", {})
            .setdefault("memory", {})
            .setdefault("eval", {})
        )
        mem["use_triton_triangle_kernels"] = False

    _patch_loader(monkeypatch, mutate)
