"""The eight protein-only RFD4 condition families, through the CLI that serves them.

Each family is `tt-bio design --model rfd4` on the input and settings upstream runs it with
(`perf/rfd4/cond/families.py`). These tests drive that command, with the engine replaced by a
recorder: everything before and after the card is real — the condition spec is loaded and built,
the input is featurized, the sampling arguments are resolved, and the design is written back as
an annotated CIF. What a card would add is the trunk, and that is graded by `rfd4-trunk`,
`rfd4-e2e` and the family grades in `state/rfd4/FAMILIES-PROTEIN.md`.

The inputs are upstream's own and are staged, not vendored:

    python perf/rfd4/cond/families.py stage      # RFD4_UPSTREAM=<clone>

Without them the input-bound tests skip; the surface tests do not.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
MINE = ("unconditional", "binder", "antibody_cdr", "nanobody_cdr", "enzyme", "motif_scaffold",
        "partial_diffusion", "inverse_folding")


def _families():
    spec = importlib.util.spec_from_file_location("rfd4_families",
                                                  ROOT / "perf" / "rfd4" / "cond" / "families.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("rfd4_families", module)
    spec.loader.exec_module(module)
    return module


families = _families()


class RecordingEngine:
    """Stands in for the card: records what it was asked for, hands the input straight back.

    The writer checks the design against the input's conditioning — coordinate-conditioned atoms
    must be present, atomized tokens must carry -1 — so a canned output would fail those checks
    for a reason that has nothing to do with the CLI. Echoing the input is the one answer that is
    valid for every family, and it leaves the checks doing their job.
    """

    def __init__(self):
        self.calls = []

    def design(self, features, *, num_designs, nsteps, seed, **sampling):
        import torch

        from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_ae_masks
        from tt_bio.rfd4.ae_fold import tables
        from tt_bio.rfd4.engine import DesignOutput, preprocess

        n = features["padding_mask"].shape[-1]
        self.calls.append({"tokens": int(n), "num_designs": num_designs, "nsteps": nsteps,
                           "seed": seed, "sampling": sampling})
        batch = preprocess(features)
        _, emit = compute_ae_masks(batch)
        # An emitted token with no input sequence has to come back with one: the writer refuses a
        # design that still carries mask tokens, which is exactly what a real run never produces.
        tab = tables()
        allowed = tab["allowed"][batch["token_type"].long()].clone()
        allowed[..., tab["mask_token"]] = False
        residue_type = torch.where(features["residue_type"] == tab["mask_token"],
                                   allowed.float().argmax(-1), features["residue_type"])
        residue_type = torch.where(emit, residue_type,
                                   torch.full_like(residue_type, -1))
        # The writer also checks the atom inventory against the residue types, so the mask has to
        # be the legal slots of whatever was just assigned, as the decoder's own output is.
        legal = tab["atom_mask"][residue_type.clamp(min=0)]
        atom_mask = (features["atom37_mask"] | legal) & legal & emit[..., None]
        # A token the decoder does not emit (an atomized ligand) keeps its CA slot, which is how
        # the engine reports one.
        atom_mask[..., 1] |= features["padding_mask"].bool() & ~emit

        def wide(t):
            return t.expand(num_designs, *t.shape[1:]).clone()

        return DesignOutput(coords_nm=wide(batch["coords_nm"]),
                            residue_type=wide(residue_type),
                            atom_mask=wide(atom_mask),
                            insertion_logits=torch.zeros(num_designs, n, n))


@pytest.fixture
def engine(monkeypatch):
    from tt_bio import rfd4

    recorder = RecordingEngine()
    monkeypatch.setattr(rfd4, "get_engine", lambda *a, **k: recorder)
    return recorder


def available(family):
    """The family's input, or None when the upstream inputs have not been staged."""
    f = families.FAMILIES[family]
    if "from" in f:
        return None
    path = families.input_path(family)
    staged = families.staged(family)
    if "spec" in f:
        # The spec is in the repo; the structure it points at is staged.
        return path if path.exists() and (families.INPUTS / "5o45_cropped.cif.zst").exists() else None
    return path if staged and staged.exists() else None


def run(engine, family, tmp_path, extra=()):
    from tt_bio.main import cli

    cmd = families.command(family, tmp_path / family, seed=0, num_designs=1)[1:]
    result = CliRunner().invoke(cli, cmd + list(extra))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


# ---- the table itself ----------------------------------------------------------------

def test_the_table_is_exactly_this_rows_eight_families():
    assert tuple(families.FAMILIES) == MINE


def test_every_family_names_a_recycle_schedule_the_sampler_ships():
    from tt_bio.rfd4.sampler import RECYCLE_SCHEDULES

    for name, f in families.FAMILIES.items():
        assert f["recipe"] in RECYCLE_SCHEDULES, name


def test_every_family_has_an_input_or_a_parent():
    for name, f in families.FAMILIES.items():
        assert ("input" in f) or ("spec" in f) or ("from" in f), name


# ---- the CLI surface, no input needed ------------------------------------------------

def test_partial_diffusion_and_inverse_folding_are_mutually_exclusive(tmp_path, engine):
    """Upstream refuses this combination; refusing it later, on the card, wastes a run."""
    from tt_bio.main import cli

    cif = available("inverse_folding")
    if cif is None:
        pytest.skip("upstream inputs not staged")
    result = CliRunner().invoke(cli, ["design", "--model", "rfd4", str(cif), "--annotated_cif",
                                      "--out_dir", str(tmp_path), "--inverse_folding",
                                      "--partial", "0.5"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_an_unknown_recipe_is_named_not_guessed(tmp_path, engine):
    from tt_bio.main import cli

    cif = available("unconditional")
    if cif is None:
        pytest.skip("upstream inputs not staged")
    result = CliRunner().invoke(cli, ["design", "--model", "rfd4", str(cif), "--annotated_cif",
                                      "--out_dir", str(tmp_path), "--recipe", "thinking_hard"])
    assert result.exit_code != 0
    assert "unknown recycle strategy" in result.output


# ---- one test per family, through the CLI --------------------------------------------

@pytest.mark.parametrize("family", [f for f in MINE if f != "partial_diffusion"])
def test_family_designs_through_the_cli(family, tmp_path, engine):
    if available(family) is None:
        pytest.skip(f"{family}: upstream input not staged")
    run(engine, family, tmp_path)
    call = engine.calls[-1]
    assert call["tokens"] > 0
    written = list((tmp_path / family).glob("*_generated.cif"))
    assert written, "the serving path writes a design, not just features"


def test_inverse_folding_reaches_the_engine_as_inverse_folding(tmp_path, engine):
    if available("inverse_folding") is None:
        pytest.skip("upstream input not staged")
    run(engine, "inverse_folding", tmp_path)
    assert engine.calls[-1]["sampling"]["inverse_folding"] is True


def test_partial_diffusion_restarts_from_a_design(tmp_path, engine):
    """Its input is a design, so it runs on the output of the binder run before it."""
    if available("binder") is None:
        pytest.skip("upstream input not staged")
    run(engine, "binder", tmp_path)
    parent = next((tmp_path / "binder").glob("*_generated.cif"))

    from tt_bio.main import cli

    cmd = families.command("partial_diffusion", tmp_path / "partial", seed=0, num_designs=1,
                           parent_design=parent)[1:]
    result = CliRunner().invoke(cli, cmd)
    assert result.exit_code == 0, result.output + str(result.exception)
    frac = engine.calls[-1]["sampling"]["partial_trajectory_fraction_steps"]
    assert frac == pytest.approx(0.4166666, rel=1e-5)
    assert list((tmp_path / "partial").glob("*_generated.cif"))
