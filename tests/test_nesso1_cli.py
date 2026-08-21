"""Host-only contract tests for the `tt-bio affinity` CLI wiring.

Pins: the command exists and accepts nesso1; nesso1 is deliberately NOT a predict
--model choice (it folds nothing, so it must not ride the structure-writing scheduler
path the seven fold models share); the seed default is on, because upstream is not
reproducible run to run. No device, no checkpoint, no network.
"""
from __future__ import annotations

from click.testing import CliRunner


def test_affinity_command_accepts_nesso1():
    from tt_bio.main import AFFINITY_MODELS, affinity_cmd

    assert "nesso1" in AFFINITY_MODELS
    model_opt = next(p for p in affinity_cmd.params if p.name == "model")
    assert list(model_opt.type.choices) == list(AFFINITY_MODELS)


def test_nesso1_is_not_a_predict_model():
    """predict writes structures, PAE/PDE and confidence and fans jobs across a
    scheduler built for that. Nesso-1 returns a scalar. Keeping it out of
    PREDICT_MODELS also keeps release_gate/perf_regression from deriving coverage
    for a model they cannot fold."""
    from tt_bio.main import PREDICT_MODELS, predict

    assert "nesso1" not in PREDICT_MODELS
    model_opt = next(p for p in predict.params if p.name == "model")
    assert "nesso1" not in model_opt.type.choices


def test_seed_defaults_to_repeatable():
    """The featurizer roto-translates every conformer off the global torch RNG, so
    upstream differs run to run (64/64 affinity values, max 0.058). Our default pins
    it; -1 is the opt-out."""
    from tt_bio.main import affinity_cmd
    from tt_bio.nesso1 import DEFAULT_SEED

    seed_opt = next(p for p in affinity_cmd.params if p.name == "seed")
    assert seed_opt.default is None  # resolved to DEFAULT_SEED in the body
    assert isinstance(DEFAULT_SEED, int)


def test_help_lists_the_outputs():
    from tt_bio.main import affinity_cmd

    out = CliRunner().invoke(affinity_cmd, ["--help"])
    assert out.exit_code == 0
    for token in ("affinity.csv", "_affinity.json", "--trunk", "--num_workers"):
        assert token in out.output, token


def test_multiple_cards_are_refused(tmp_path):
    """Nesso-1 is batch-1 by construction, so --devices 0,1 is a user error rather
    than something to silently ignore."""
    from tt_bio.main import affinity_cmd

    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text("version: 1\n")
    out = CliRunner().invoke(affinity_cmd, [str(yaml_path), "--devices", "0,1"])
    assert out.exit_code != 0
    assert "batch-1" in out.output
