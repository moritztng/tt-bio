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


def test_one_record_that_raises_does_not_end_the_screen(tmp_path, monkeypatch):
    """A device failure on one ligand is that ligand's error row; the rest still score."""
    import types

    import torch

    import tt_bio.nesso1 as n1
    import tt_bio.nesso1_input as inp

    ids = ["a", "big", "c"]
    manifest = types.SimpleNamespace(records=[types.SimpleNamespace(id=i) for i in ids])
    dataset = [{"id": i} for i in ids]
    monkeypatch.setattr(inp, "prepare", lambda *a, **k: (dataset, manifest, ["x_bad"]))
    monkeypatch.setattr(inp, "collate", lambda item: {"id": item["id"], "token_pad_mask": torch.ones(1, 8)})

    class Model:
        predict_args: dict = {}

        def predict(self, feats):
            if feats["id"] == "big":
                raise RuntimeError("TT_FATAL @ bank_manager.cpp:439: false\nOut of Memory")
            return {"affinity_pred_value": torch.tensor([0.5]),
                    "affinity_probability_binary": torch.tensor([0.25])}

    monkeypatch.setattr(n1.Nesso1, "from_pretrained", classmethod(lambda cls, *a, **k: Model()))
    seen = []
    rows = n1.screen(tmp_path, tmp_path / "out", use_tenstorrent=False, progress=seen.append)
    by_id = {r["id"]: r for r in rows}
    assert by_id["a"]["affinity_pred_value"] == 0.5 and by_id["c"]["affinity_pred_value"] == 0.5
    assert by_id["big"]["error"].startswith("RuntimeError: ")
    assert by_id["x_bad"]["error"] == "could not be parsed"
    assert [r["id"] for r in seen] == ["a", "big", "c"]
