"""Unit coverage for the designability harvester (scripts/boltzgen_designability.py).

Device-free: exercises the CSV-parsing / column-selection / pass-rate logic
against synthetic ``aggregate_metrics_analyze.csv`` tables, so a regression in
the harvest (wrong column picked, bad pass-rate math) is caught in CI without
running the on-device design pipeline. The on-device numbers themselves live in
docs/boltzgen-designability.md; here we only pin the reduction.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "boltzgen_designability.py"


def _load():
    spec = importlib.util.spec_from_file_location("boltzgen_designability", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_metrics(dirpath: Path, rows: dict) -> Path:
    sub = dirpath / "intermediate_designs_inverse_folded"
    sub.mkdir(parents=True)
    csv = sub / "aggregate_metrics_analyze.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return dirpath


def test_prefers_isolation_refold_column(tmp_path):
    """designfolding-bb_rmsd (standalone refold) wins over bb_rmsd_design."""
    mod = _load()
    out = _write_metrics(tmp_path, {
        "id": ["a", "b"],
        "designed_sequence": ["AAAA", "CCCCCC"],
        "designfolding-bb_rmsd": [0.6, 3.5],
        "bb_rmsd_design": [9.9, 9.9],  # deliberately wrong: must not be chosen
    })
    res = mod.score(out, sc_threshold=2.0)
    assert res["column"] == "designfolding-bb_rmsd"
    assert res["min"] == pytest.approx(0.6)
    assert res["max"] == pytest.approx(3.5)
    # 1 of 2 <= 2A strict; 2 of 2 <= 4A permissive.
    assert res["pass_strict"] == pytest.approx(0.5)
    assert res["pass_permissive"] == pytest.approx(1.0)


def test_falls_back_to_complex_refold(tmp_path):
    """Protocols without design_folding expose only bb_rmsd_design."""
    mod = _load()
    out = _write_metrics(tmp_path, {
        "id": ["a", "b", "c", "d"],
        "designed_sequence": ["AA", "AA", "AA", "AA"],
        "bb_rmsd_design": [1.0, 1.5, 2.5, 5.0],
    })
    res = mod.score(out, sc_threshold=2.0)
    assert res["column"] == "bb_rmsd_design"
    assert res["pass_strict"] == pytest.approx(0.5)   # 1.0, 1.5 <= 2
    assert res["pass_permissive"] == pytest.approx(0.75)  # all but 5.0 <= 4
    assert res["median"] == pytest.approx(2.0)


def test_pass_threshold_tracks_custom_bar(tmp_path):
    mod = _load()
    out = _write_metrics(tmp_path, {
        "id": ["a", "b", "c", "d"],
        "designed_sequence": ["A", "A", "A", "A"],
        "designfolding-bb_rmsd": [0.5, 1.0, 2.9, 3.1],
    })
    res = mod.score(out, sc_threshold=3.0)
    assert res["pass_threshold"] == pytest.approx(0.75)  # <=3.0: 0.5,1.0,2.9


def test_missing_column_exits(tmp_path):
    mod = _load()
    out = _write_metrics(tmp_path, {"id": ["a"], "designed_sequence": ["A"],
                                    "some_other_metric": [1.0]})
    with pytest.raises(SystemExit):
        mod.score(out, sc_threshold=2.0)


def test_no_csv_exits(tmp_path):
    mod = _load()
    with pytest.raises(SystemExit):
        mod.score(tmp_path, sc_threshold=2.0)
