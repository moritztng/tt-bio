"""`--diffusion_precision` reaches the Protenix build, keys the worker's reload, and is reported on
the models that do not read it. Host only: the dispatch and the model build are stubbed, and any
device open raises instead of opening a chip.

The option exists because bf16 diffusion is the one Protenix-v2 precision lever that buys time on
Wormhole. It ships off because it can pick a different binding mode where fp32 is uncertain
(docs/protenix-diffusion-precision.md):
unset, every config keeps the fp32 build and the reload hash it had before the option existed.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from tt_bio import host_controller as H
from tt_bio.capabilities import unread_flags
from tt_bio.main import cli
from tt_bio.runtime import build_local_workers

PROTENIX = {"model": "protenix-v2", "fast": False}


@pytest.fixture(autouse=True)
def _restore_fast_mode(monkeypatch):
    # `load_model` sets the process-wide fast mode from its config; a `fast: True` case left
    # it on and every later host test read bfloat8_b as the device dtype.
    import tt_bio.tenstorrent as T
    monkeypatch.setattr(T, "_FAST_MODE", T._FAST_MODE)


def test_unset_keeps_the_hash_it_had_before():
    assert H.run_config_hash({**PROTENIX, "diffusion_precision": None}) == H.run_config_hash(PROTENIX)


def test_bf16_is_a_different_warm_model():
    assert H.run_config_hash({**PROTENIX, "diffusion_precision": "bf16"}) != H.run_config_hash(PROTENIX)


def test_only_protenix_reads_it():
    assert unread_flags("protenix-v2", {"--diffusion_precision": True}) == []
    notes = unread_flags("boltz2", {"--diffusion_precision": True})
    assert len(notes) == 1 and "ignores --diffusion_precision" in notes[0]


@pytest.mark.parametrize("prec,want", [(None, None), ("fp32", True), ("bf16", False)])
def test_worker_builds_the_requested_precision(monkeypatch, prec, want):
    import tt_bio.protenix as P
    import tt_bio.tenstorrent as T
    from tt_bio import worker as W

    monkeypatch.setattr(T, "get_device", lambda *a, **k: pytest.fail("opened a device"))
    seen = {}

    def fake_load(path, **kw):
        seen.update(kw, path=path)
        return object()

    monkeypatch.setattr(P.Protenix, "load_from_checkpoint", staticmethod(fake_load))
    state = W._WorkerState("tenstorrent")
    state.load_model({**PROTENIX, "protenix_ckpt": "ck.pt", "diffusion_precision": prec})
    assert seen == {"path": "ck.pt", "diffusion_fp32": want}


@pytest.mark.parametrize("args,want", [([], None), (["--diffusion_precision", "bf16"], "bf16")])
def test_cli_puts_it_in_the_run_config(monkeypatch, tmp_path, args, want):
    import tt_bio.main as m
    import tt_bio.tenstorrent as T

    monkeypatch.setattr(T, "get_device", lambda *a, **k: pytest.fail("opened a device"))
    monkeypatch.setattr(m, "download_all", lambda *a, **k: None)
    monkeypatch.setattr(m, "_local_workers", lambda *a, **k: build_local_workers("cpu", [object()], [0]))
    got = {}

    def grab(run_payload, workers, **kw):
        got.update(run_payload["config"])
        return 0

    monkeypatch.setattr(m, "_dispatch_run", grab)
    y = tmp_path / "t.yaml"
    y.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKVLAAGIVG\n")
    r = CliRunner().invoke(cli, ["predict", str(y), "--model", "protenix-v2", "--accelerator",
                                 "tenstorrent", "--single_sequence", "--out_dir", str(tmp_path / "o"),
                                 *args])
    assert r.exit_code == 0, r.output
    assert got["diffusion_precision"] == want


def test_cli_refuses_an_unknown_precision(tmp_path):
    r = CliRunner().invoke(cli, ["predict", str(tmp_path), "--diffusion_precision", "fp16"])
    assert r.exit_code == 2 and "fp16" in r.output


@pytest.mark.parametrize("model,guided,prec,want", [
    ("opendde", True, None, True), ("opendde-abag", True, None, True),
    ("opendde", True, "bf16", False), ("opendde", False, None, None)])
def test_guided_opendde_builds_fp32_under_fast(monkeypatch, model, guided, prec, want):
    # A guided run re-docks the movable chains on a clashing x0; bf16 diffusion compressed them
    # ~9x as often as upstream, so guidance resolves to fp32 in every mode unless asked otherwise,
    # and the warm-model key carries it so a resident bf16 --fast model is not reused.
    import tt_bio.opendde as O
    import tt_bio.tenstorrent as T
    from tt_bio import worker as W

    monkeypatch.setattr(T, "get_device", lambda *a, **k: pytest.fail("opened a device"))
    seen = {}
    monkeypatch.setattr(O.OpenDDE, "load_from_checkpoint",
                        staticmethod(lambda path, **kw: seen.update(kw) or object()))
    cfg = {"model": model, "fast": True, "opendde_ckpt": "ck.pt", "use_tfg_guidance": guided,
           "diffusion_precision": prec}
    W._WorkerState("tenstorrent").load_model(cfg)
    assert seen["diffusion_fp32"] is want
    unguided = {**cfg, "use_tfg_guidance": False}
    assert (H.run_config_hash(cfg) != H.run_config_hash(unguided)) == (guided and prec is None)
