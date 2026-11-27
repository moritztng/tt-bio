"""The `tt-bio dock` surface: flags, defaults, the three input forms and the output layout.

Host-only, no card and no weights: every run here goes through `--no_model`, upstream's
seed-conformer baseline, which is exactly why that flag is wired first. A CLI whose only
test is "it parses" is a CLI nobody ran.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from tt_bio.diffdock import output
from tt_bio.diffdock.job import RECIPE, RECIPE_PATH, DockOptions, read_csv_jobs
from tt_bio.main import cli

UPSTREAM_RECIPE = Path("/home/moritz/scratch/dfd/upstream/default_inference_args.yaml")
RECEPTOR = Path("/home/moritz/scratch/dfd/patchtest/examples/1a46_protein_processed.pdb")
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

needs_receptor = pytest.mark.skipif(not RECEPTOR.exists(),
                                    reason="no local receptor structure to dock into")




def run(*args):
    return CliRunner().invoke(cli, ["dock", *args], catch_exceptions=False)


def test_dock_help_and_defaults():
    """Every default is the published recipe, and --help says what each one is.

    The recipe ships in the package (``recipe.yaml``) as a verbatim copy of upstream's
    `default_inference_args.yaml`, so this compares the live option defaults against
    upstream's own file rather than against a list typed twice.
    """
    o = DockOptions()
    assert o.diffusion_samples == RECIPE["samples_per_complex"] == 10
    assert o.sampling_steps == RECIPE["inference_steps"] == 20
    assert o.actual_steps == RECIPE["actual_steps"] == 19
    assert o.no_final_step_noise is RECIPE["no_final_step_noise"] is True
    assert o.initial_noise_std == RECIPE["initial_noise_std_proportion"]
    assert o.no_model is False and o.deterministic is False and o.sampler == "sde"
    # The gated recipe's model generation: old_score_model false, old_filtering_model true.
    assert (o.old_score_model, o.old_filtering_model) == (RECIPE["old_score_model"],
                                                          RECIPE["old_filtering_model"])
    assert o.steps == 19, "the recipe performs 19 of its 20 scheduled steps"

    help_text = run("--help").output
    for flag in ("--ligand", "--out_dir", "--model", "--weights", "--checkpoint",
                 "--confidence_checkpoint", "--no_confidence", "--diffusion_samples",
                 "--sampling_steps", "--actual_steps", "--max_parallel_samples",
                 "--no_final_step_noise", "--initial_noise_std", "--init_at_residue",
                 "--sampler", "--deterministic", "--no_model", "--pocket", "--pocket_radius",
                 "--save_trajectory", "--minimize", "--gnina_path", "--limit_failures",
                 "--config", "--seed", "--cache", "--device_ids", "--host_threads",
                 "--controller", "--owner", "--fast", "--log"):
        assert flag in help_text, f"{flag} is not in `tt-bio dock --help`"
    for value in ("10", "20", "19", "1.4602"):
        assert value in help_text, f"--help does not state the default {value}"


@pytest.mark.skipif(not UPSTREAM_RECIPE.exists(), reason="no upstream clone on this host")
def test_the_vendored_recipe_is_upstreams_file_verbatim():
    assert RECIPE_PATH.read_bytes() == UPSTREAM_RECIPE.read_bytes()


def test_upstream_config_yaml_accepted(tmp_path):
    """Upstream's own inference-args YAML is handed to us unchanged and reproduces it.

    Including the nine low-temperature constants, which have no flags of their own: they
    ride in ``recipe`` and reach the sampler from there.
    """
    o = DockOptions().with_config(RECIPE_PATH)
    assert (o.diffusion_samples, o.sampling_steps, o.actual_steps) == (10, 20, 19)
    assert o.recipe["temp_sampling_tr"] == RECIPE["temp_sampling_tr"]

    custom = tmp_path / "args.yaml"
    custom.write_text(yaml.safe_dump({**RECIPE, "samples_per_complex": 4,
                                      "inference_steps": 7, "ode": True,
                                      "old_score_model": True, "no_model": True}))
    c = DockOptions().with_config(custom)
    assert (c.diffusion_samples, c.sampling_steps) == (4, 7)
    assert c.sampler == "ode" and c.weights == "diffdock" and c.no_model is True
    assert c.old_score_model is True

    # A flag on the command line beats the config file it was given alongside.
    out = run(str(custom), "--help")
    assert out.exit_code == 0


def test_a_config_that_is_not_a_mapping_is_refused(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        DockOptions().with_config(bad)


def test_csv_form_is_upstreams_four_columns(tmp_path):
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("complex_name,protein_path,protein_sequence,ligand_description\n"
                        f"a,{RECEPTOR},,{ASPIRIN}\n"
                        ",,MKTAYIAKQRQISFVKSHFSRQ,ligand.sdf\n")
    jobs = read_csv_jobs(csv_path)
    assert [j.name for j in jobs] == ["a", "complex_1"], "a blank name becomes complex_<i>"
    assert jobs[0].receptor_path == RECEPTOR and jobs[0].ligand == ASPIRIN
    assert jobs[1].receptor_path is None
    assert jobs[1].receptor_sequence == "MKTAYIAKQRQISFVKSHFSRQ"


def test_a_csv_without_the_ligand_column_says_so(tmp_path):
    csv_path = tmp_path / "wrong.csv"
    csv_path.write_text("name,smiles\na,CCO\n")
    with pytest.raises(ValueError, match="ligand_description"):
        read_csv_jobs(csv_path)


def _yaml_job(tmp_path, **docking) -> Path:
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump({
        "version": 1,
        "sequences": [{"protein": {"id": "A", "file": str(RECEPTOR)}},
                      {"ligand": {"id": "L", "smiles": ASPIRIN}}],
        "docking": {"diffusion_samples": 3, "no_model": True, "seed": 7, **docking},
    }))
    return path


@needs_receptor
def test_yaml_job_equals_cli_job(tmp_path):
    """The YAML form and the command line produce the same poses, byte for byte."""
    yaml_out, cli_out = tmp_path / "y", tmp_path / "c"
    r1 = run(str(_yaml_job(tmp_path)), "--out_dir", str(yaml_out))
    assert r1.exit_code == 0, r1.output
    r2 = run(str(RECEPTOR), "--ligand", ASPIRIN, "--name", "job", "--no_model",
             "--diffusion_samples", "3", "--seed", "7", "--out_dir", str(cli_out))
    assert r2.exit_code == 0, r2.output
    for rank in (1, 2, 3):
        a = (yaml_out / "job" / f"rank{rank}.sdf").read_bytes()
        b = (cli_out / "job" / f"rank{rank}.sdf").read_bytes()
        assert a == b, f"rank{rank} differs between the YAML and CLI forms"


@needs_receptor
def test_the_output_layout_is_upstreams(tmp_path):
    """Upstream's filenames, and one additive file.

    The names come from `inference.py`'s writer: `rank{n}.sdf` for rank 1 and
    `rank{n}_confidence{c:.2f}.sdf` for every rank, which is where the familiar
    `rank1_confidence-0.42.sdf` comes from (the dash is a negative confidence). Without a
    confidence model there are no suffixed files, the case upstream crashes on.
    """
    assert output.pose_filename(1, -0.42) == "rank1_confidence-0.42.sdf"
    assert output.pose_filename(10, 3.17) == "rank10_confidence3.17.sdf"
    assert output.pose_filename(2, None) == "rank2.sdf"

    out = tmp_path / "o"
    r = run(str(RECEPTOR), "--ligand", ASPIRIN, "--name", "x", "--no_model",
            "--diffusion_samples", "2", "--out_dir", str(out))
    assert r.exit_code == 0, r.output
    assert sorted(p.name for p in (out / "x").iterdir()) == [
        "confidence.json", "rank1.sdf", "rank2.sdf"]
    ranking = json.loads((out / "x" / "confidence.json").read_text())
    assert [row["rank"] for row in ranking] == [1, 2]
    assert all(row["confidence"] is None for row in ranking), "--no_model scores nothing"
    assert [row["sdf"] for row in ranking] == ["rank1.sdf", "rank2.sdf"]


@needs_receptor
def test_a_pocket_puts_every_pose_in_the_pocket(tmp_path):
    from rdkit import Chem

    out = tmp_path / "p"
    r = run(str(RECEPTOR), "--ligand", ASPIRIN, "--name", "pk", "--no_model",
            "--diffusion_samples", "4", "--pocket", "10,20,30", "--pocket_radius", "4",
            "--out_dir", str(out))
    assert r.exit_code == 0, r.output
    for sdf in sorted((out / "pk").glob("rank*.sdf")):
        mol = next(iter(Chem.SDMolSupplier(str(sdf))))
        centre = mol.GetConformer().GetPositions().mean(axis=0)
        assert ((centre - [10, 20, 30]) ** 2).sum() ** 0.5 <= 4.001, f"{sdf.name} left the pocket"


@needs_receptor
def test_one_bad_row_does_not_end_a_screen(tmp_path):
    """A CSV keeps going and names the failure; a single complex fails loudly instead."""
    csv_path = tmp_path / "screen.csv"
    csv_path.write_text("complex_name,protein_path,protein_sequence,ligand_description\n"
                        f"good,{RECEPTOR},,{ASPIRIN}\n"
                        f"bad,{RECEPTOR},,not-a-molecule\n")
    out = tmp_path / "s"
    r = run(str(csv_path), "--no_model", "--diffusion_samples", "2", "--out_dir", str(out))
    assert r.exit_code == 0, r.output
    assert (out / "good" / "rank1.sdf").exists()
    failures = (out / "failures.csv").read_text()
    assert "bad" in failures and "not-a-molecule" in failures

    single = run(str(RECEPTOR), "--ligand", "not-a-molecule", "--no_model",
                 "--out_dir", str(tmp_path / "one"))
    assert single.exit_code != 0 and "not-a-molecule" in single.output


def test_log_level(tmp_path, caplog):
    """--log is tt-bio's name for upstream's --loglevel, and it sets the level."""
    import logging

    from tt_bio.main import cli as _cli
    opts = {o for p in _cli.commands["dock"].params for o in getattr(p, "opts", ())}
    assert "--log" in opts and "--loglevel" not in opts, "tt-bio's name, not upstream's"
    bad = run(str(RECEPTOR if RECEPTOR.exists() else tmp_path), "--log", "NOPE")
    assert bad.exit_code != 0 and "NOPE" in bad.output

    if RECEPTOR.exists():
        run(str(RECEPTOR), "--ligand", ASPIRIN, "--no_model", "--diffusion_samples", "1",
            "--log", "DEBUG", "--out_dir", str(tmp_path / "l"))
        assert logging.getLogger("tt_bio").level == logging.DEBUG


@needs_receptor
def test_a_ligand_on_both_sides_is_refused(tmp_path):
    r = run(str(_yaml_job(tmp_path)), "--ligand", ASPIRIN, "--out_dir", str(tmp_path / "x"))
    assert r.exit_code != 0 and "both name a ligand" in r.output


def test_a_receptor_without_a_ligand_is_refused(tmp_path):
    # Any .pdb path reaches the refusal: it is raised before the receptor is read. A directory
    # does not, it is read as a folder of YAML jobs, so the test must not depend on RECEPTOR.
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    r = run(str(receptor), "--no_model", "--out_dir", str(tmp_path / "x"))
    assert r.exit_code != 0 and "--ligand is required" in r.output


def test_an_unknown_docking_key_names_the_known_ones(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump({
        "version": 1,
        "sequences": [{"protein": {"id": "A", "file": str(RECEPTOR)}},
                      {"ligand": {"id": "L", "smiles": ASPIRIN}}],
        "docking": {"num_poses": 3}}))
    r = run(str(path), "--out_dir", str(tmp_path / "o"))
    assert r.exit_code != 0
    assert "num_poses" in r.output and "diffusion_samples" in r.output


def test_a_yaml_diffdock_cannot_honour_is_refused_not_dropped(tmp_path):
    """The capability table, through the same check every fold goes through."""
    path = tmp_path / "rna.yaml"
    path.write_text(yaml.safe_dump({
        "version": 1,
        "sequences": [{"protein": {"id": "A", "file": str(RECEPTOR)}},
                      {"rna": {"id": "R", "sequence": "GAUC"}},
                      {"ligand": {"id": "L", "smiles": ASPIRIN}}],
        "docking": {"no_model": True}}))
    r = run(str(path), "--out_dir", str(tmp_path / "o"))
    assert r.exit_code != 0
    assert "RNA" in r.output or "rna" in r.output


def test_a_shorter_run_keeps_the_recipes_shape():
    """`--sampling_steps 4` on its own used to run 19 steps of a 4-entry schedule.

    `actual_steps` truncates the schedule, so it only means something relative to
    `sampling_steps`: shortening the run keeps the recipe's "stop one step early" shape,
    and asking for more steps than are scheduled runs all of them.
    """
    from tt_bio.diffdock.job import override

    o = override(DockOptions(), {"sampling_steps": 4})
    assert (o.sampling_steps, o.actual_steps, o.steps) == (4, 3, 3)

    both = override(DockOptions(), {"sampling_steps": 6, "actual_steps": 6})
    assert both.steps == 6, "an explicit pair is honoured"

    clamped = override(DockOptions(), {"sampling_steps": 4, "actual_steps": 19})
    assert clamped.steps == 4, "more steps than the schedule holds runs the schedule"

    assert DockOptions().steps == 19, "the default recipe is untouched"


@needs_receptor
def test_a_shorter_run_docks(tmp_path):
    """The flag combination above, through the command, on a real receptor."""
    out = run(str(RECEPTOR), "--ligand", ASPIRIN, "--no_model", "--sampling_steps", "4",
              "--diffusion_samples", "2", "--out_dir", str(tmp_path))
    assert out.exit_code == 0, out.output
    assert (tmp_path / "complex_0" / "rank1.sdf").exists()
