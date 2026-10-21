"""The two halves of `rfd4-evaluate` a user reaches by hand: opt-in metrics and offline scoring.

Upstream keeps the physics metrics out of `all_inline_metrics.yaml` and the Foldseek novelty /
clustering behind `FOLDSEEK_EXEC` plus a database, so no `--metrics_tags` value switches them on.
tt-bio does not take those dependencies (state/rfd4/DECISIONS.md), so naming one is refused with
the dependency it would need. The other half is the part that IS ported: scoring a folder of
structures that already exist, with a `task_csv` giving each one its own task tag.
"""
import csv
import shutil
from pathlib import Path

import pytest

from tt_bio.rfd4.evaluate import OPT_IN_NOT_PORTED, evaluate

GOLDENS = Path(__file__).resolve().parents[1] / "perf" / "rfd4" / "eval_goldens"
DESIGNS = ("clean_generated.cif", "clashy_generated.cif")


def _folder(tmp_path):
    """A folder of structures that already exist on disk, which is what offline scoring takes."""
    inputs = tmp_path / "existing"
    inputs.mkdir()
    for name in DESIGNS:
        shutil.copy(GOLDENS / name, inputs / name)
    return inputs


@pytest.mark.parametrize("name", sorted(OPT_IN_NOT_PORTED))
def test_an_optin_metric_is_refused_with_the_dependency_it_needs(tmp_path, name):
    with pytest.raises(NotImplementedError) as excinfo:
        evaluate(_folder(tmp_path), "design", tmp_path / "out", extra_metrics=[name])
    message = str(excinfo.value)
    assert name in message and OPT_IN_NOT_PORTED[name].split(",")[0] in message


def test_the_optin_table_covers_upstreams_physics_and_foldseek_metrics():
    """The eight tmol presets, the QM IPA metric, the PyRosetta salt one and Foldseek novelty."""
    assert sum(1 for dep in OPT_IN_NOT_PORTED.values() if dep.startswith("tmol")) == 8
    assert {"ipa_interaction_energy", "charge_complementarity", "foldseek_novelty"} <= set(OPT_IN_NOT_PORTED)


def test_foldseek_clustering_is_refused_rather_than_silently_dropped(tmp_path):
    with pytest.raises(NotImplementedError, match="Foldseek"):
        evaluate(_folder(tmp_path), "design", tmp_path / "out", cluster_with_foldseek=True)


def test_an_unknown_metric_name_is_named_not_ignored(tmp_path):
    with pytest.raises(ValueError, match="not_a_metric"):
        evaluate(_folder(tmp_path), "design", tmp_path / "out", extra_metrics=["not_a_metric"])


def test_a_folder_of_existing_structures_scores_without_a_generation_run(tmp_path):
    """Offline scoring: the inputs are structures on disk, not designs this process produced."""
    out = evaluate(_folder(tmp_path), "design", tmp_path / "out")
    rows = {Path(r["designed_structure_path"]).name: r for r in csv.DictReader(open(out))}
    assert set(rows) == set(DESIGNS)
    # The clashy fixture is the clean one with residues 40-60 pulled in, so it is the one that
    # has to come out different; a run that scored both the same would pass a smoke test.
    clean = float(rows["clean_generated.cif"]["clash_metrics.frac_atoms_clashing_inter_residue"])
    clashy = float(rows["clashy_generated.cif"]["clash_metrics.frac_atoms_clashing_inter_residue"])
    assert clean == 0.0 and clashy > 0.1


def test_task_csv_tags_each_structure_with_its_own_task(tmp_path):
    inputs = _folder(tmp_path)
    task_csv = tmp_path / "tasks.csv"
    with open(task_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["designed_structure_path", "task"])
        writer.writerow([str(inputs / "clean_generated.cif"), "keepers"])
    out = evaluate(inputs, "design", tmp_path / "out", task_csv=task_csv)
    tasks = {Path(r["designed_structure_path"]).name: r["task"] for r in csv.DictReader(open(out))}
    assert tasks == {"clean_generated.cif": "keepers", "clashy_generated.cif": "all"}
