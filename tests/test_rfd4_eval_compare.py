"""`perf/rfd4/eval_compare.py` is what every "exact vs upstream" claim on the evaluate port rests on,
so its failure modes are tested rather than assumed."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "rfd4_eval_compare", Path(__file__).resolve().parents[1] / "perf" / "rfd4" / "eval_compare.py"
)
cmp_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["rfd4_eval_compare"] = cmp_mod
_SPEC.loader.exec_module(cmp_mod)


def write_csv(path, rows, columns):
    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


BASE = {
    "example_id": "d_0_generated_abc12345",
    "designed_structure_path": "/somewhere/d_0_generated.cif",
    "clash_metrics.frac_atoms_clashing_1p5A": "0.0",
    "sse_distribution.helix_fraction": "0.86",
    "bond_length_metrics.bond_length.frac_unreasonable.between_chain.polymer_polymer": "",
}
COLUMNS = list(BASE)


def run(tmp_path, upstream_row, port_row):
    up = write_csv(tmp_path / "up.csv", [upstream_row], COLUMNS)
    pt = write_csv(tmp_path / "pt.csv", [port_row], list(port_row))
    return cmp_mod.compare(cmp_mod.load(up), cmp_mod.load(pt), atol=0.0, rtol=0.0)


def test_identical_is_exact(tmp_path):
    assert run(tmp_path, dict(BASE), dict(BASE))["exact"]


def test_example_id_may_differ_because_the_join_is_on_the_path(tmp_path):
    port = dict(BASE, example_id="d_0")
    assert run(tmp_path, dict(BASE), port)["exact"]


def test_value_difference_is_caught(tmp_path):
    port = dict(BASE, **{"sse_distribution.helix_fraction": "0.85"})
    report = run(tmp_path, dict(BASE), port)
    assert not report["exact"]
    assert "sse_distribution.helix_fraction" in report["designs"]["d_0_generated.cif"]["diffs"]


def test_a_missing_column_is_a_mismatch_not_a_skip(tmp_path):
    port = {k: v for k, v in BASE.items() if k != "sse_distribution.helix_fraction"}
    report = run(tmp_path, dict(BASE), port)
    assert not report["exact"]
    assert report["missing_from_port"] == ["sse_distribution.helix_fraction"]


def test_empty_cell_is_not_zero(tmp_path):
    col = "bond_length_metrics.bond_length.frac_unreasonable.between_chain.polymer_polymer"
    port = dict(BASE, **{col: "0.0"})
    assert not run(tmp_path, dict(BASE), port)["exact"]


def test_nan_matches_nan(tmp_path):
    port = dict(BASE, **{"sse_distribution.helix_fraction": "nan"})
    upstream = dict(BASE, **{"sse_distribution.helix_fraction": "nan"})
    assert run(tmp_path, upstream, port)["exact"]


def test_tolerance_is_opt_in(tmp_path):
    up = write_csv(tmp_path / "up.csv", [dict(BASE)], COLUMNS)
    port_row = dict(BASE, **{"sse_distribution.helix_fraction": "0.8600000001"})
    pt = write_csv(tmp_path / "pt.csv", [port_row], COLUMNS)
    assert not cmp_mod.compare(cmp_mod.load(up), cmp_mod.load(pt), atol=0.0, rtol=0.0)["exact"]
    assert cmp_mod.compare(cmp_mod.load(up), cmp_mod.load(pt), atol=1e-6, rtol=0.0)["exact"]


def test_a_design_on_one_side_only_is_not_exact(tmp_path):
    up = write_csv(tmp_path / "up.csv", [dict(BASE)], COLUMNS)
    other = dict(BASE, designed_structure_path="/somewhere/e_0_generated.cif")
    pt = write_csv(tmp_path / "pt.csv", [dict(BASE), other], COLUMNS)
    report = cmp_mod.compare(cmp_mod.load(up), cmp_mod.load(pt), atol=0.0, rtol=0.0)
    assert not report["exact"]
    assert report["designs_port_only"] == ["e_0_generated.cif"]


def test_duplicate_design_is_rejected(tmp_path):
    path = write_csv(tmp_path / "dup.csv", [dict(BASE), dict(BASE)], COLUMNS)
    with pytest.raises(ValueError, match="duplicate design"):
        cmp_mod.load(path)
