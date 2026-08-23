"""The nesso1 perf cell measures `tt-bio affinity`, not a Boltz-2 fold.

`_measure_affinity` serves two models. boltz2-affinity folds a structure and then scores it
(`predict --model boltz2` with five sampling knobs); nesso1 folds nothing and ignores every one
of those knobs (`affinity --model nesso1` at CLI defaults). One function, two commands, chosen by
the model name.

A merge dropped that branch once, along with SPECS["nesso1"] itself, and the only thing that
noticed was `test_perf_model_coverage.py` -- which fails on the missing SPECS entry and says
nothing about which command the cell would have run. These tests pin the command and the row, so
a nesso1 cell that quietly starts timing a Boltz-2 fold, or records a sampling protocol it never
ran, fails on the host with no card.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _perf_regression():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "tt_bio_perf_regression_cell", REPO / "scripts" / "perf_regression.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _capture(monkeypatch, model):
    """Run _measure_affinity with the timing stubbed, and return (commands, result row)."""
    mod = _perf_regression()
    seen = []

    def fake_time_single_shot(make_cmd, env, work, timeout, label):
        for rep in range(2):
            cmd, log = make_cmd(rep)
            seen.append([str(x) for x in cmd])
        return 1.0, [1.0, 1.0]

    monkeypatch.setattr(mod, "_time_single_shot", fake_time_single_shot)
    monkeypatch.setattr(mod, "detect_card_type", lambda *a, **k: "p150a")
    out = Path(work_dir(monkeypatch)) / "row.json"
    mod._measure_affinity(model, mod.SPECS[model], out)
    import json
    return seen, json.loads(out.read_text())


def work_dir(monkeypatch):
    import tempfile
    return tempfile.mkdtemp(prefix="perf-cell-test-")


@pytest.mark.parametrize("model", ["nesso1", "boltz2-affinity"])
def test_the_cell_exists_at_all(model):
    """Both affinity cells are in SPECS. The nesso1 entry has been silently dropped once."""
    mod = _perf_regression()
    assert model in mod.SPECS, f"SPECS has no {model} cell"
    assert mod.SPECS[model]["kind"] == "affinity"


def test_nesso1_times_the_affinity_verb(monkeypatch):
    cmds, row = _capture(monkeypatch, "nesso1")
    assert cmds, "no command was built"
    for cmd in cmds:
        assert "affinity" in cmd, cmd
        assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "nesso1", cmd
        assert "predict" not in cmd, "the nesso1 cell must not time a Boltz-2 fold"
        # Nesso-1 has no diffusion and no structure sampling; passing these would either be
        # rejected by the CLI or, worse, silently ignored and recorded as the protocol.
        for knob in ("--sampling_steps", "--diffusion_samples", "--recycling_steps",
                     "--sampling_steps_affinity", "--diffusion_samples_affinity",
                     "--single_sequence", "--affinity_mw_correction"):
            assert knob not in cmd, f"{knob} does not apply to nesso1"
    assert row["unit"] == "affinities/s"
    # A row that carries sampling knobs claims a protocol this model never ran.
    for knob in ("sampling_steps", "diffusion_samples", "recycling_steps",
                 "sampling_steps_affinity", "diffusion_samples_affinity"):
        assert knob not in row, f"nesso1 row records {knob}, which it does not use"
    assert "bf16 trunk" in row["input"] and "CLI defaults" in row["input"]


def test_boltz2_affinity_still_times_the_fold(monkeypatch):
    """The other model through the same function is unchanged."""
    cmds, row = _capture(monkeypatch, "boltz2-affinity")
    assert cmds
    for cmd in cmds:
        assert "predict" in cmd, cmd
        assert cmd[cmd.index("--model") + 1] == "boltz2", cmd
        assert "--affinity_mw_correction" in cmd
        assert "--sampling_steps_affinity" in cmd
    assert row["unit"] == "affinities/s"
    assert row["sampling_steps"] and row["diffusion_samples_affinity"]
    assert "single-seq, affinity mode" in row["input"]


def test_the_two_cells_read_the_same_fixture():
    """The comparison in docs/nesso1.md only holds if both cells fold the same input."""
    mod = _perf_regression()
    assert mod.AFFINITY.exists(), f"missing affinity fixture {mod.AFFINITY}"
    assert mod.SPECS["nesso1"]["kind"] == mod.SPECS["boltz2-affinity"]["kind"] == "affinity"
