"""The ported RFD4 evaluate metrics against upstream's own output.

`perf/rfd4/eval_goldens/upstream_design_metrics.json` is what upstream's `rfd4-evaluate` wrote for
the two CIFs beside it, in the reference venv at the pinned sha. Upstream rounds the CSV to 4
decimals and writes NaN as an empty cell, so that is the comparison this test makes.

`clean_generated.cif` is a CPU fp32 RFD4 design and scores zero everywhere, which only proves the
port raises no false positives. `clashy_generated.cif` is that design with residues 40-60 pulled in
towards the centroid: 19.73 % of its atoms clash and two backbone bonds are stretched past the
1.8 A threshold, so it is the case that can actually tell a wrong implementation apart.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import filter_amino_acids, filter_nucleotides
from biotite.structure.io.pdbx import CIFFile, get_structure

from tt_bio.rfd4.eval_metrics import compute_bond_length_stats, compute_clash_stats

GOLDENS = Path(__file__).resolve().parents[1] / "perf" / "rfd4" / "eval_goldens"
UPSTREAM = json.loads((GOLDENS / "upstream_design_metrics.json").read_text())


def port_columns(cif_path):
    atoms = get_structure(CIFFile.read(str(cif_path)), model=1, include_bonds=True)
    is_polymer = filter_amino_acids(atoms) | filter_nucleotides(atoms)
    out = {f"bond_length_metrics.{k}": v for k, v in compute_bond_length_stats(atoms, is_polymer).items()}
    out.update({f"clash_metrics.{k}": v for k, v in compute_clash_stats(atoms, is_polymer).items()})
    return out


def as_upstream_cell(value):
    """Upstream's CSV goes through DataFrame.round(4); NaN lands as an empty cell."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{round(float(value), 4):g}"


@pytest.mark.parametrize("design", sorted(UPSTREAM["rows"]))
def test_port_matches_upstream_design_metrics(design):
    expected = UPSTREAM["rows"][design]
    ours = port_columns(GOLDENS / design)
    compared = 0
    for column, upstream_value in expected.items():
        if not column.startswith(("clash_metrics.", "bond_length_metrics.")):
            continue  # the rest of the design tag is not ported yet
        assert column in ours, f"{column} missing from the port"
        got = as_upstream_cell(ours[column])
        want = "" if upstream_value in ("", None) else f"{float(upstream_value):g}"
        assert got == want, f"{column}: port {got!r} vs upstream {upstream_value!r}"
        compared += 1
    assert compared == 11, f"expected 11 ported columns, compared {compared}"


def test_the_clashy_fixture_actually_discriminates():
    """A fixture that scores zero everywhere cannot catch a wrong implementation."""
    clashy = UPSTREAM["rows"]["clashy_generated.cif"]
    assert float(clashy["clash_metrics.frac_atoms_clashing_1p5A"]) > 0.1
    assert float(clashy["bond_length_metrics.bond_length.frac_unreasonable"]) > 0


def test_no_bonds_means_no_bond_length_columns():
    """Upstream returns {} rather than zeros, which is how the cell ends up empty."""
    atoms = get_structure(CIFFile.read(str(GOLDENS / "clean_generated.cif")), model=1, include_bonds=False)
    is_polymer = filter_amino_acids(atoms) | filter_nucleotides(atoms)
    assert compute_bond_length_stats(atoms, is_polymer) == {}


def test_an_empty_bond_category_is_nan_not_zero():
    """A monomer has no between-chain bonds; upstream reports nan there, never 0.0."""
    ours = port_columns(GOLDENS / "clean_generated.cif")
    for column in (
        "bond_length_metrics.bond_length.frac_unreasonable.between_chain.polymer_polymer",
        "bond_length_metrics.bond_length.frac_unreasonable.between_chain.nonpolymer_to_polymer",
    ):
        assert np.isnan(ours[column])


def test_evaluate_writes_upstreams_design_csv(tmp_path):
    """The runner end to end: every column upstream wrote for tag `design`, cell for cell."""
    import csv
    import shutil

    from tt_bio.rfd4.evaluate import evaluate

    inputs = tmp_path / "designs"
    inputs.mkdir()
    for name in UPSTREAM["rows"]:
        shutil.copy(GOLDENS / name, inputs / name)
    out = evaluate(inputs, "[design]", tmp_path / "out")
    rows = {r["designed_structure_path"].rsplit("/", 1)[-1]: r for r in csv.DictReader(open(out))}
    for design, expected in UPSTREAM["rows"].items():
        ours = rows[design]
        for column, want in expected.items():
            assert column in ours, f"{design}: {column} missing"
            got = ours[column]
            if want in ("", None) or got == "":
                assert got == (want or ""), f"{design} {column}: {got!r} vs {want!r}"
            else:
                assert float(got) == float(want), f"{design} {column}: {got!r} vs {want!r}"


def test_a_tag_that_needs_an_unported_metric_is_refused(tmp_path):
    from tt_bio.rfd4.evaluate import evaluate

    with pytest.raises(NotImplementedError, match="hotspot_obedience"):
        evaluate(GOLDENS, "design,hotspot_conditioning", tmp_path)


def test_partial_diffusion_tag_matches_upstream(tmp_path):
    """ca_rmsd_to_input runs the same comparison core as every refolding RMSD (mask per side,
    Kabsch on the align atoms, RMSD on the subset), so this pins that core too: 3.7611 A here."""
    import csv
    import shutil

    from tt_bio.rfd4.evaluate import evaluate

    golden = json.loads((GOLDENS / "upstream_partial_diffusion.json").read_text())
    inputs = tmp_path / "pd"
    shutil.copytree(GOLDENS / "partial_diffusion", inputs / "partial_diffusion")
    out = evaluate(inputs, "partial_diffusion", tmp_path / "out")
    rows = {r["designed_structure_path"].rsplit("/", 1)[-1]: r for r in csv.DictReader(open(out))}
    assert golden["rows"]["pd_generated.cif"]["ca_rmsd_to_input.rmsd"] == "3.7611"
    for design, expected in golden["rows"].items():
        for column, want in expected.items():
            if column == "example_id":
                continue  # hashes the path relative to INPUTS, which differs between the two runs
            got = rows[design][column]
            assert (got == want) if "" in (got, want) else float(got) == float(want), (design, column, got, want)
