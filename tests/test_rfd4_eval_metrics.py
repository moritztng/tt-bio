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

    with pytest.raises(NotImplementedError, match="rasa_adherence"):
        evaluate(GOLDENS, "design,rasa_conditioning", tmp_path)


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


def test_unconditional_refolding_tag_matches_upstream(tmp_path):
    """The oracle-bound half, off the refold cache upstream itself resumes from.

    `perf/rfd4/eval_goldens/refold/` is that cache for `clean_generated.cif`: a refold (the design
    rotated, translated and jittered, `perf/rfd4/make_refold_fixture.py`) and the confidences
    `tt_bio/rfd4/oracle.py` reduces for it. Upstream scored exactly these files in the reference
    venv with `metrics_tags=[design,unconditional_refolding]`, no AF3 anywhere, and
    `upstream_unconditional_refolding.json` is what it wrote.
    """
    import csv
    import shutil

    from tt_bio.rfd4.evaluate import evaluate

    golden = json.loads((GOLDENS / "upstream_unconditional_refolding.json").read_text())
    inputs = tmp_path / "designs"
    inputs.mkdir()
    shutil.copy(GOLDENS / "clean_generated.cif", inputs / "clean_generated.cif")
    out = tmp_path / "out"
    shutil.copytree(GOLDENS / "refold", out / "evaluation_samples")

    csv_path = evaluate(inputs, "design,unconditional_refolding", out)
    rows = {r["designed_structure_path"].rsplit("/", 1)[-1]: r for r in csv.DictReader(open(csv_path))}
    expected = golden["rows"]["clean_generated.cif"]
    # 1.5663 A against a 1.5 A bar: the criterion's failing branch, and a refold-dependent number
    # that a wrong alignment or a wrong backbone mask cannot reproduce by accident.
    assert expected["refolded_backbone_rmsd.rmsd"] == "1.5663"
    assert expected["unconditional.pass"] == "0.0"
    compared = 0
    for column, want in expected.items():
        if column in ("example_id", "dataset_name", "designed_structure_path", "output_base_dir") or column.endswith(
            ("refolded_structure_path", "confidence_json_path", "designed_chain_path")
        ):
            continue  # provenance: names the out_dir, which differs between the two runs
        got = rows["clean_generated.cif"][column]
        if column in ("task", "metrics_tags", "chain_of_interest"):
            assert got == want, (column, got, want)  # chain_of_interest is the chain iid, `A_1`
        else:
            assert (got == want) if "" in (got, want) else float(got) == float(want), (column, got, want)
        compared += 1
    assert compared >= 45, f"expected the whole row compared, got {compared} columns"


def test_ppi_refolding_tag_matches_upstream(tmp_path):
    """The binder half: a two-chain design with a conditioned target, scored through `ppi_refolding`.

    `perf/rfd4/eval_goldens/ppi/binder_generated.cif` is `clean_generated.cif` split at residue 50,
    the second half moved clear and handed back as a fixed target (`perf/rfd4/make_ppi_fixture.py`),
    which is what makes chain A the unambiguous chain of interest and `sa_mask('C_CRD')` non-empty.
    Both arms read the same cached refold. The row covers what the unconditional tag cannot: the two
    target-aligned binder RMSDs, the per-chain-pair oracle keys, the COI reductions, and a `ppi`
    criterion that passes on all three of its thresholds.
    """
    import csv
    import shutil

    from tt_bio.rfd4.evaluate import evaluate

    golden = json.loads((GOLDENS / "upstream_ppi_refolding.json").read_text())
    inputs = tmp_path / "designs"
    inputs.mkdir()
    shutil.copy(GOLDENS / "ppi" / "binder_generated.cif", inputs / "binder_generated.cif")
    out = tmp_path / "out"
    (out / "evaluation_samples").mkdir(parents=True)
    shutil.copytree(
        GOLDENS / "ppi" / "refold_binder_generated_975e6ee5",
        out / "evaluation_samples" / "binder_generated_975e6ee5",
    )

    csv_path = evaluate(inputs, "design,ppi_refolding", out)
    rows = {r["designed_structure_path"].rsplit("/", 1)[-1]: r for r in csv.DictReader(open(csv_path))}
    expected = golden["rows"]["binder_generated.cif"]
    assert expected["refolded_target_backbone_aligned_binder_rmsd.rmsd"] == "1.3824"
    assert expected["oracle_ppi.min_ipae_A_1_B_1"] == "0.4006"
    assert expected["ppi.pass"] == "1.0"
    compared = 0
    for column, want in expected.items():
        if column in ("example_id", "dataset_name", "designed_structure_path", "output_base_dir") or column.endswith(
            ("refolded_structure_path", "confidence_json_path", "designed_chain_path")
        ):
            continue
        got = rows["binder_generated.cif"][column]
        if column in ("task", "metrics_tags", "chain_of_interest"):
            assert got == want, (column, got, want)
        else:
            assert (got == want) if "" in (got, want) else float(got) == float(want), (column, got, want)
        compared += 1
    assert compared >= 53, f"expected the whole row compared, got {compared} columns"


def _assert_row_matches(expected, got_row, min_compared):
    compared = 0
    for column, want in expected.items():
        if column in ("example_id", "dataset_name", "designed_structure_path", "output_base_dir") or column.endswith(
            ("refolded_structure_path", "confidence_json_path", "designed_chain_path")
        ):
            continue
        got = got_row[column]
        if column in ("task", "metrics_tags", "chain_of_interest"):
            assert got == want, (column, got, want)
        else:
            assert (got == want) if "" in (got, want) else float(got) == float(want), (column, got, want)
        compared += 1
    assert compared >= min_compared, f"expected the whole row compared, got {compared} columns"


def test_hotspot_conditioning_tag_matches_upstream(tmp_path):
    """`hotspot_obedience` on a two-chain design whose halves stay in contact.

    `make_ppi_fixture.py --in-contact --hotspots ...` marks 101 atoms on 12 target residues, of
    which 19 have a generated binder atom within 4.5 A: 0.1881, neither branch of the trivial 0/1.
    The same row also grades `clash_metrics` on a real inter-chain interface (0.0025).
    """
    import csv
    import shutil

    from tt_bio.rfd4.evaluate import evaluate

    golden = json.loads((GOLDENS / "upstream_hotspot.json").read_text())
    inputs = tmp_path / "designs"
    inputs.mkdir()
    shutil.copy(GOLDENS / "hotspot" / "hotspot_generated.cif", inputs / "hotspot_generated.cif")
    csv_path = evaluate(inputs, "design,hotspot_conditioning", tmp_path / "out")
    rows = {r["designed_structure_path"].rsplit("/", 1)[-1]: r for r in csv.DictReader(open(csv_path))}
    expected = golden["rows"]["hotspot_generated.cif"]
    assert expected["hotspot_obedience.hotspot_obedience"] == "0.1881"
    _assert_row_matches(expected, rows["hotspot_generated.cif"], 34)


def _run_cached(tmp_path, design, refold_dir, eid, tags):
    import csv
    import shutil

    from tt_bio.rfd4.evaluate import evaluate

    inputs = tmp_path / "designs"
    inputs.mkdir()
    shutil.copy(design, inputs / design.name)
    out = tmp_path / "out"
    (out / "evaluation_samples").mkdir(parents=True)
    shutil.copytree(refold_dir, out / "evaluation_samples" / eid)
    csv_path = evaluate(inputs, tags, out)
    return {r["designed_structure_path"].rsplit("/", 1)[-1]: r for r in csv.DictReader(open(csv_path))}


def test_equivalent_atom_flips_are_resolved_as_upstream(tmp_path):
    """A refold that hands each of 57 equivalent pairs (ASP OD1/OD2, PHE ring edges, ...) the
    other's coordinate (`make_refold_fixture.py --flip-equivalent`). Upstream permutes the design's
    labels to match before any RMSD; without that step the target-aligned binder RMSD reads 1.0387
    instead of 0.5314."""
    golden = json.loads((GOLDENS / "upstream_symmetry_flip.json").read_text())
    rows = _run_cached(
        tmp_path,
        GOLDENS / "ppi" / "binder_generated.cif",
        GOLDENS / "symmetry" / "refold_flip_binder_generated_975e6ee5",
        "binder_generated_975e6ee5",
        "design,ppi_refolding",
    )
    expected = golden["rows"]["binder_generated.cif"]
    assert expected["refolded_target_aligned_binder_rmsd.rmsd"] == "0.5314"
    _assert_row_matches(expected, rows["binder_generated.cif"], 53)


def test_swapped_identical_chains_are_resolved_as_upstream(tmp_path):
    """A homodimer whose refold has its two chains exchanged and its equivalent atoms flipped
    (`make_ppi_fixture.py --homodimer`, `make_refold_fixture.py --swap-chains A,B`). Upstream
    matches the copies first; without that the backbone RMSD reads 18.21 instead of 0.5207."""
    golden = json.loads((GOLDENS / "upstream_symmetry_dimer.json").read_text())
    rows = _run_cached(
        tmp_path,
        GOLDENS / "symmetry" / "dimer_generated.cif",
        GOLDENS / "symmetry" / "refold_dimer_generated_0e3de0a0",
        "dimer_generated_0e3de0a0",
        "design,unconditional_refolding",
    )
    expected = golden["rows"]["dimer_generated.cif"]
    assert expected["refolded_backbone_rmsd.rmsd"] == "0.5207"
    _assert_row_matches(expected, rows["dimer_generated.cif"], 51)


def test_motif_scaffold_refolding_matches_upstream_on_a_real_design(tmp_path):
    """The motif RMSD pair and the `motif_scaffold` criterion on a real upstream RFD4 design.

    `motif/pdl1_binder_0_generated.cif` is the PD-L1 binder from the reference env, whose target is
    the coordinate-conditioned motif (`SA_CRDSRC`, `coords-with-insertion`). The cached refold
    flips 100 equivalent pairs, so the per-group symmetry step fires too."""
    golden = json.loads((GOLDENS / "upstream_motif_real_binder.json").read_text())
    rows = _run_cached(
        tmp_path,
        GOLDENS / "motif" / "pdl1_binder_0_generated.cif",
        GOLDENS / "motif" / "refold_pdl1_binder_0_generated_479647c2",
        "pdl1_binder_0_generated_479647c2",
        "design,motif_scaffold_refolding",
    )
    expected = golden["rows"]["pdl1_binder_0_generated.cif"]
    assert expected["refolded_motif_allatom_rmsd.rmsd"] == "0.5367"
    assert expected["refolded_motif_ca_rmsd.rmsd"] == "0.5486"
    assert expected["motif_scaffold.pass"] == "1.0"
    _assert_row_matches(expected, rows["pdl1_binder_0_generated.cif"], 53)


def test_motif_family_tags_match_upstream_on_a_real_design(tmp_path):
    """The campaign's motif family, `design,motif_scaffold_refolding,3d_coordinate_conditioning`,
    which adds the inline `design_motif_allatom_rmsd` (the design against its own conditioning)."""
    golden = json.loads((GOLDENS / "upstream_motif_family_real_binder.json").read_text())
    rows = _run_cached(
        tmp_path,
        GOLDENS / "motif" / "pdl1_binder_0_generated.cif",
        GOLDENS / "motif" / "refold_pdl1_binder_0_generated_479647c2",
        "pdl1_binder_0_generated_479647c2",
        "design,motif_scaffold_refolding,3d_coordinate_conditioning",
    )
    expected = golden["rows"]["pdl1_binder_0_generated.cif"]
    assert expected["design_motif_allatom_rmsd.rmsd"] == "0.178"
    _assert_row_matches(expected, rows["pdl1_binder_0_generated.cif"], 54)


@pytest.mark.parametrize(
    "golden_name, tags, criterion",
    [
        ("upstream_ame_real_binder.json", "design,ame_refolding,motif_scaffold_refolding", "ame"),
        ("upstream_amenl_real_binder.json", "design,ame_refolding,motif_scaffold_refolding,no_ligand", "ame_no_ligand"),
    ],
)
def test_enzyme_family_criteria_match_upstream(tmp_path, golden_name, tags, criterion):
    """`ame` (prohibited by `no_ligand`) and `ame_no_ligand` (requires it), on the real design."""
    golden = json.loads((GOLDENS / golden_name).read_text())
    rows = _run_cached(
        tmp_path,
        GOLDENS / "motif" / "pdl1_binder_0_generated.cif",
        GOLDENS / "motif" / "refold_pdl1_binder_0_generated_479647c2",
        "pdl1_binder_0_generated_479647c2",
        tags,
    )
    expected = golden["rows"]["pdl1_binder_0_generated.cif"]
    assert expected[f"{criterion}.pass"] == "1.0"
    _assert_row_matches(expected, rows["pdl1_binder_0_generated.cif"], 55)
