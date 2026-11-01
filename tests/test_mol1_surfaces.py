"""Mol-1's upstream commands and runner options, each mapped onto tt-bio and checked without a card.

`mol pred` options -> `tt-bio predict --model mol1`; `--load_checkpoint_path` -> `--checkpoint`; first-run download ->
the weights registry, fetched before any worker starts; `mol doctor` -> `tt-bio preflight`; `--device` -> `--devices`,
where a card that is not there is an error and a CPU/GPU request is refused rather than run on the card.
The dispatch layer is stubbed (tests/test_predict_exit_code.py's fixture).
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

import tt_bio.main as m
from tests.test_predict_exit_code import _target, host_predict  # noqa: F401 (fixture)
from tt_bio import runtime, weights
from tt_bio.main import cli

REAL_DOWNLOAD_ALL = m.download_all
REAL_LOCAL_WORKERS = m._local_workers


def _predict(tmp_path, *extra):
    return CliRunner().invoke(cli, ["predict", str(_target(tmp_path / "t.yaml")), "--model", "mol1",
                                    "--single_sequence", "--out_dir", str(tmp_path / "out"), *extra])


@pytest.fixture
def fetched(monkeypatch, host_predict):
    """Records every weights row predict fetches, in order, and the config dispatch receives."""
    got = {"keys": [], "cfg": None}
    monkeypatch.setattr(m, "download_all", REAL_DOWNLOAD_ALL)
    monkeypatch.setattr(weights, "fetch", lambda key, root=None, **_: got["keys"].append(key))

    def dispatch(payload, workers, *, total, **_):
        host_predict["total"] = total
        got["cfg"] = payload["config"]
        return 0

    monkeypatch.setattr(m, "_dispatch_run", dispatch)
    return got


def test_the_checkpoint_and_the_ccd_library_download_before_the_first_fold(fetched, tmp_path):
    result = _predict(tmp_path)
    assert result.exit_code == 0, result.output
    assert {"mol1", "mols"} <= set(fetched["keys"])
    assert fetched["cfg"]["checkpoint"] is None


def test_the_checkpoint_source_is_verified_and_has_a_fallback():
    art = weights.ARTIFACTS["mol1"]
    assert art.sha256 == "d5b83a86884aa6870565b1a249e8f13c232dfc6dfa311089ab0c7868212b6bd6"
    assert len(art.sources) == 2 and art.sources[1].startswith("https://mol-1-release.")
    assert {"mol1", "mols"} <= {a.key for a in weights.artifacts_for("mol1")}


def test_checkpoint_flag_replaces_the_download(fetched, tmp_path):
    """upstream --load_checkpoint_path: the given file is folded and the shipped one is not fetched."""
    ckpt = tmp_path / "mine.pt"
    ckpt.write_bytes(b"x")
    result = _predict(tmp_path, "--checkpoint", str(ckpt))
    assert result.exit_code == 0, result.output
    assert "does not read --checkpoint" not in result.output
    assert "mol1" not in fetched["keys"] and "mols" in fetched["keys"]
    assert fetched["cfg"]["checkpoint"] == str(ckpt.resolve())


def test_the_worker_loads_the_given_checkpoint(monkeypatch, tmp_path):
    from tt_bio import worker

    monkeypatch.setattr(weights, "cache_root", lambda *a: tmp_path)
    monkeypatch.setattr(weights, "fetch", lambda key, root=None, **_: tmp_path / key)
    cfg = {"model": "mol1", "checkpoint": "/data/mine.pt"}
    worker._ensure_local_artifacts(cfg)
    assert cfg["protenix_ckpt"] == "/data/mine.pt"
    cfg = {"model": "mol1"}
    worker._ensure_local_artifacts(cfg)
    assert cfg["protenix_ckpt"] == str(tmp_path / "mol1")


@pytest.mark.parametrize("ready", [True, False])
def test_preflight_reports_whether_mol1_can_run(monkeypatch, tmp_path, ready):
    """`mol doctor`'s question for tt-bio: are the weights here and intact, and if not can this host reach them."""
    stats = [weights.Status(weights.ARTIFACTS[k], "present" if ready else "missing", tmp_path / k, 0)
             for k in ("mol1", "mols")]
    monkeypatch.setattr(weights, "model_ready", lambda model, root=None: (ready, stats))
    monkeypatch.setattr(weights, "probe", lambda url: {"ok": True, "host": url.split("/")[2], "bytes": 2048,
                                                         "seconds": 0.1, "mb_per_s": 1.0})
    result = CliRunner().invoke(cli, ["preflight", "mol1", "--cache", str(tmp_path)])
    if ready:
        assert result.exit_code == 0 and "READY" in result.output, result.output
    else:
        assert result.exit_code == 1 and "MISSING" in result.output, result.output
        assert "tt-bio preflight mol1 --download" in result.output
        assert "mol-1-release.oss-cn-shanghai.aliyuncs.com" in result.output


def test_preflight_notes_a_missing_kalign_without_failing(monkeypatch, tmp_path):
    """kalign only matters for --use_template with .a3m/.hhr hits, so its absence is a note, not a red."""
    stats = [weights.Status(weights.ARTIFACTS[k], "present", tmp_path / k, 0) for k in ("mol1", "mols")]
    monkeypatch.setattr(weights, "model_ready", lambda model, root=None: (True, stats))
    monkeypatch.setattr(m.shutil, "which", lambda name: None if name == "kalign" else "/bin/" + name)
    result = CliRunner().invoke(cli, ["preflight", "mol1", "--cache", str(tmp_path)])
    assert result.exit_code == 0 and "needs kalign" in result.output, result.output
    monkeypatch.setattr(m.shutil, "which", lambda name: "/bin/" + name)
    assert "kalign" not in CliRunner().invoke(cli, ["preflight", "mol1", "--cache", str(tmp_path)]).output


@pytest.mark.parametrize("accelerator", ["cpu", "gpu"])
def test_a_device_mol1_cannot_use_is_refused_not_substituted(host_predict, tmp_path, accelerator):
    result = _predict(tmp_path, "--accelerator", accelerator)
    assert result.exit_code == 2 and "Tenstorrent only" in result.output
    assert host_predict["total"] is None


def test_an_absent_card_is_an_error_naming_the_cards_present(host_predict, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_local_workers", REAL_LOCAL_WORKERS)
    monkeypatch.setattr(m, "_require_ttnn", lambda: None)
    monkeypatch.setattr(runtime.glob, "glob", lambda pat: ["/dev/tenstorrent/0", "/dev/tenstorrent/1"])
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    result = _predict(tmp_path, "--devices", "7")
    assert result.exit_code != 0 and host_predict["total"] is None
    assert "[7] not available (present: 0, 1)" in str(result.exception or result.output)

# `mol pred` option -> the `tt-bio predict` flag that carries it. Options with no flag are DECISIONS.md D-io-1/D-io-2
# (CUDA execution switches, --use_default_params) and D-io-3 (comma seeds -> modelSeeds + --use_seeds_in_json).
MOL_PRED = {"--input": "DATA", "--out_dir": "--out_dir", "--seeds": "--seed", "--cycle": "--recycling_steps",
            "--step": "--sampling_steps", "--sample": "--diffusion_samples", "--model_name": "--model",
            "--load_checkpoint_path": "--checkpoint", "--use_msa": "--single_sequence",
            "--use_template": "--use_template", "--use_rna_msa": "--use_rna_msa",
            "--use_seeds_in_json": "--use_seeds_in_json", "--need_atom_confidence": "--write_pae",
            "--device": "--devices"}


def test_every_mol_pred_option_has_a_predict_flag():
    out = CliRunner().invoke(cli, ["predict", "--help"]).output
    missing = [up for up, ours in MOL_PRED.items() if ours not in out]
    assert not missing, missing


def test_mol1_defaults_are_upstreams():
    """`mol pred` defaults: -c 20, -p 200, -e 5, seed 101 for a JSON job."""
    assert m.RECYCLING_STEPS["mol1"] == 20
    assert m.SAMPLING_STEPS.get("mol1", m.SAMPLING_STEPS_DEFAULT) == 200
    assert m.DIFFUSION_SAMPLES["mol1"] == 5
