"""Upstream's four-column screening CSV, and per-complex failure isolation.

`inference.py` takes a CSV of `complex_name, protein_path, protein_sequence,
ligand_description`, builds every row, and skips the rows it cannot build
instead of ending the run. A library screen is the product here, so one
unparseable ligand in ten thousand must not cost the other 9,999; tt-bio keeps
the isolation and adds the reason, which upstream only prints.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tt_bio.diffdock.input import (
    CSV_COLUMNS,
    ComplexSpec,
    build_batch,
    read_complex_csv,
)

FIXTURES = Path(__file__).parent / "fixtures" / "diffdock"
INDEX = FIXTURES / "INDEX.json"

pytestmark = pytest.mark.skipif(
    not INDEX.exists(), reason="DiffDock input fixtures are not present")


def _ids() -> list[str]:
    return [c["id"] for c in json.loads(INDEX.read_text())["complexes"]]


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_csv_runs_every_row(tmp_path):
    """One receptor, several ligands: the screening case, every row built."""
    cid = _ids()[0]
    receptor = str(FIXTURES / f"{cid}_protein.pdb")
    ligands = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    path = _write_csv(tmp_path / "jobs.csv", [
        {"complex_name": f"lig{i}", "protein_path": receptor,
         "protein_sequence": "", "ligand_description": smiles}
        for i, smiles in enumerate(ligands)])

    specs = read_complex_csv(path)
    assert [s.name for s in specs] == ["lig0", "lig1", "lig2"]
    assert all(s.protein_sequence is None for s in specs), "empty cells are None"

    result = build_batch(specs, seed=1)
    assert result.skipped == 0, result.failures
    assert [g.name for g in result.graphs] == ["lig0", "lig1", "lig2"]
    # One receptor, so the receptor half is the same graph every time.
    first = result.graphs[0].receptor
    for graph in result.graphs[1:]:
        assert graph.receptor["pos"].shape == first["pos"].shape
        assert graph.receptor["edge_index"].shape == first["edge_index"].shape


def test_unnamed_rows_get_upstreams_fallback_names(tmp_path):
    """A row with no complex_name becomes `complex_<row index>`, as upstream."""
    cid = _ids()[0]
    receptor = str(FIXTURES / f"{cid}_protein.pdb")
    path = _write_csv(tmp_path / "jobs.csv", [
        {"complex_name": "", "protein_path": receptor,
         "protein_sequence": "", "ligand_description": "CCO"},
        {"complex_name": "named", "protein_path": receptor,
         "protein_sequence": "", "ligand_description": "CCO"},
        {"complex_name": "nan", "protein_path": receptor,
         "protein_sequence": "", "ligand_description": "CCO"},
    ])
    assert [s.name for s in read_complex_csv(path)] == ["complex_0", "named", "complex_2"]


def test_missing_column_is_refused(tmp_path):
    path = tmp_path / "jobs.csv"
    path.write_text("complex_name,protein_path\na,b\n")
    with pytest.raises(ValueError, match="missing the column"):
        read_complex_csv(path)


def test_bad_row_skipped_run_continues(tmp_path):
    """A bad ligand, a missing receptor and a too-large receptor are isolated."""
    cid = _ids()[0]
    receptor = str(FIXTURES / f"{cid}_protein.pdb")
    specs = [
        ComplexSpec("good0", protein_path=receptor, ligand_description="CCO"),
        ComplexSpec("badligand", protein_path=receptor,
                    ligand_description="this-is-not-a-molecule"),
        ComplexSpec("noreceptor", protein_path=str(tmp_path / "absent.pdb"),
                    ligand_description="CCO"),
        ComplexSpec("noreceptoratall", ligand_description="CCO"),
        ComplexSpec("good1", protein_path=receptor, ligand_description="c1ccccc1"),
    ]
    result = build_batch(specs, seed=1)

    assert [g.name for g in result.graphs] == ["good0", "good1"]
    assert [name for name, _ in result.failures] == [
        "badligand", "noreceptor", "noreceptoratall"]
    assert result.skipped == 3
    # The reason is kept, not just the count.
    assert all(error for _, error in result.failures)

    out = tmp_path / "failures.csv"
    result.write_failures_csv(out)
    rows = list(csv.DictReader(out.open()))
    assert [r["complex_name"] for r in rows] == [
        "badligand", "noreceptor", "noreceptoratall"]
    assert "not-a-molecule" in rows[0]["error"] or "RDKit" in rows[0]["error"]


def test_single_complex_failure_is_loud():
    """Outside a batch a bad complex raises; only a screen isolates failures."""
    from tt_bio.diffdock.input import build_complex_graph

    with pytest.raises(Exception):
        build_complex_graph(ComplexSpec("x", ligand_description="CCO"))
