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


def test_multiple_cards_are_refused(tmp_path, monkeypatch):
    """Nesso-1 is batch-1 by construction, so --devices 0,1 is a user error rather
    than something to silently ignore.

    TT_VISIBLE_DEVICES has to be cleared: affinity_cmd reads --devices only when the env
    var is unset (the env wins, deliberately), and RELEASING.md tells you to run the suite
    as `TT_VISIBLE_DEVICES=0 pytest`, so this failed under the documented invocation and
    passed under a bare one."""
    from tt_bio.main import affinity_cmd

    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)

    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text("version: 1\n")
    out = CliRunner().invoke(affinity_cmd, [str(yaml_path), "--devices", "0,1"])
    assert out.exit_code != 0
    assert "batch-1" in out.output


class TestTheFleetPath:
    """`tt-bio affinity` had no --controller, so the one dispatch route a served
    platform has could not reach it: production runs every job kind through the shared
    controller because the persistent workers already hold the local devices, and a
    second local process cannot open them. Without this the model is CLI-only."""

    def test_the_option_exists_and_carries_owner(self):
        from tt_bio.main import affinity_cmd

        names = {p.name for p in affinity_cmd.params}
        assert {"controller", "owner"} <= names

    def test_the_worker_serves_it_through_the_affinity_path(self):
        """Discovered from AFFINITY_MODELS, not spelled out again in worker.py."""
        from tt_bio.main import AFFINITY_MODELS
        from tt_bio.worker import _is_affinity_model, _is_embed_model

        for model in AFFINITY_MODELS:
            assert _is_affinity_model(model)
            assert not _is_embed_model(model)
        assert not _is_affinity_model("boltz2")

    def test_the_model_is_resident_across_leases(self):
        """The cold load is 12.2 s against a 1.3 s warm forward, so loading per job
        would make the fleet path slower than the local one it replaces."""
        import inspect

        from tt_bio.worker import _WorkerState

        assert "_is_affinity_model" in inspect.getsource(_WorkerState.load_model)
        assert "nesso1.load" in inspect.getsource(_WorkerState.load_model)

    def test_both_paths_write_the_same_files(self):
        """A fleet run and a local run have to leave the same output behind, or the
        platform reads one shape and a CLI user the other."""
        import inspect

        from tt_bio.main import affinity_cmd

        src = inspect.getsource(affinity_cmd.callback)
        assert src.count("_write_affinity_csv") == 2

    def test_dispatch_refuses_an_empty_controller(self, tmp_path):
        from click.testing import CliRunner

        (tmp_path / "in").mkdir()
        out = CliRunner().invoke(
            __import__("tt_bio.main", fromlist=["affinity_cmd"]).affinity_cmd,
            [str(tmp_path / "in"), "--controller", "http://127.0.0.1:1", "--out_dir",
             str(tmp_path / "out")])
        assert out.exit_code != 0
        assert "No .yaml inputs" in out.output
