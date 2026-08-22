"""Regression: a killed download must never be cached as if it were complete.

Every site that resolved a weight file used to gate on ``path.exists()`` alone, so a
download killed mid-flight (worker respawn, watchdog, SIGKILL, network blip) left a
truncated multi-GB file that was then reused forever, surfacing as
``PytorchStreamReader ... failed finding central directory``. pc had a live instance of
this when the registry was written: ``~/.boltz/boltzgen/boltz2_aff.ckpt`` was 1887072256
bytes against the repo's 2061914091.

These tests pin the four properties that make that impossible, and the one property that
makes the fix safe to roll out: an already-populated cache is adopted, never re-fetched.
``perf/weights-unify/repro_poisoning.py`` demonstrates the same five sites against real
artifacts.
"""
from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from tt_bio import weights
from tt_bio.main import DESIGN_MODELS, EMBED_MODELS, PREDICT_MODELS, SAPROT_MODELS


def _zip(path: Path, n: int = 4) -> Path:
    """A valid archive standing in for a .ckpt/.pt (both are zips)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for i in range(n):
            z.writestr(f"data/{i}", b"x" * 4096)
    return path


def _truncate(path: Path, frac: float = 0.6) -> Path:
    with open(path, "r+b") as fh:
        fh.truncate(int(path.stat().st_size * frac))
    return path


# --------------------------------------------------------------------------
# The registry is the single source of truth
# --------------------------------------------------------------------------

def test_every_shipped_model_has_artifacts():
    """Anything that needs "every artifact we ship" imports the registry, so a model
    missing from it would silently get no prefetch, no status row and no docs entry."""
    for model in (*PREDICT_MODELS, *EMBED_MODELS, *SAPROT_MODELS, *DESIGN_MODELS):
        assert model in weights.MODEL_ARTIFACTS, f"{model} has no registry row"
        assert weights.artifacts_for(model), model


def test_env_overrides_are_unique_and_complete():
    """Every row takes an override, named mechanically from its key so it cannot drift,
    and the four names that existed before the registry still work."""
    canonical = [a.env for a in weights.ARTIFACTS.values()]
    assert len(canonical) == len(set(canonical))
    legacy = {v for a in weights.ARTIFACTS.values() for v in a.legacy_env}
    assert {"PROTENIX_CKPT", "OF3_CKPT", "RF3_CKPT", "OPENDDE_CKPT"} <= legacy


def test_legacy_override_wins_over_canonical(tmp_path, monkeypatch):
    """A host already exporting $OF3_CKPT must keep working unchanged."""
    ckpt = _zip(tmp_path / "of3.pt")
    monkeypatch.setenv("OF3_CKPT", str(ckpt))
    assert weights.fetch("openfold3") == ckpt


def test_artifacts_for_rejects_unknown_model():
    with pytest.raises(KeyError):
        weights.artifacts_for("no-such-model")


# --------------------------------------------------------------------------
# Integrity: the check that replaces .exists()
# --------------------------------------------------------------------------

def test_intact_rejects_truncated_archives(tmp_path):
    good = _zip(tmp_path / "good.ckpt")
    assert weights.artifact_intact(good)
    assert not weights.artifact_intact(_truncate(_zip(tmp_path / "bad.ckpt")))


def test_intact_rejects_tar_truncated_at_the_end(tmp_path):
    """Reading only the first header would pass a tar cut off at the end, which is
    exactly the shape an interrupted download leaves."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(20):
        (src / f"f{i}").write_bytes(b"y" * 8192)
    tar = tmp_path / "lib.tar"
    with tarfile.open(tar, "w") as t:
        t.add(src, arcname="lib")
    assert weights.artifact_intact(tar)
    assert not weights.artifact_intact(_truncate(tar, 0.5))


def test_intact_rejects_broken_json(tmp_path):
    (tmp_path / "m.json").write_text('{"a": 1}')
    assert weights.artifact_intact(tmp_path / "m.json")
    (tmp_path / "bad.json").write_text('{"a": ')
    assert not weights.artifact_intact(tmp_path / "bad.json")


# --------------------------------------------------------------------------
# fetch_hf_file: stage -> verify -> atomic rename
# --------------------------------------------------------------------------

def _stub_hub(monkeypatch, produce):
    """Replace hf_hub_download with `produce(staging_dir, filename) -> path`."""
    import huggingface_hub

    def fake(repo_id, filename, local_dir, force_download=False, **kw):
        return str(produce(Path(local_dir), filename))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)


def test_corrupt_cached_file_is_refetched(tmp_path, monkeypatch):
    dest = _truncate(_zip(tmp_path / "w.ckpt"))
    broken = dest.stat().st_size
    _stub_hub(monkeypatch, lambda d, f: _zip(d / f, n=8))
    out = weights.fetch_hf_file("repo", "w.ckpt", tmp_path)
    assert out == dest
    assert weights.artifact_intact(out) and out.stat().st_size != broken


def test_intact_cached_file_is_not_refetched(tmp_path, monkeypatch):
    """The property that makes this safe to deploy on hosts holding ~65 GB of weights."""
    dest = _zip(tmp_path / "w.ckpt")
    before = dest.stat().st_mtime_ns
    _stub_hub(monkeypatch, lambda d, f: pytest.fail("re-downloaded an intact artifact"))
    assert weights.fetch_hf_file("repo", "w.ckpt", tmp_path) == dest
    assert dest.stat().st_mtime_ns == before


def test_truncated_download_never_reaches_the_final_path(tmp_path, monkeypatch):
    """The core guarantee: a bad download raises and leaves the good copy in place."""
    dest = _zip(tmp_path / "w.ckpt")
    keep = dest.read_bytes()
    _stub_hub(monkeypatch, lambda d, f: _truncate(_zip(d / f)))
    with pytest.raises(RuntimeError, match="integrity check"):
        weights.fetch_hf_file("repo", "w.ckpt", tmp_path, force=True)
    assert dest.read_bytes() == keep
    assert not list(tmp_path.glob(".dl-*")), "staging left behind"


# --------------------------------------------------------------------------
# fetch_url: resumable staging, atomic rename
# --------------------------------------------------------------------------

def _stub_download(monkeypatch, produce, sizes=None):
    monkeypatch.setattr(weights, "remote_size", lambda url, timeout=15.0: (sizes or {}).get(url))
    monkeypatch.setattr(weights, "_download_to",
                        lambda url, dest, max_retries=5, quiet=False: produce(dest))


def test_url_download_is_staged_then_renamed(tmp_path, monkeypatch):
    dest = tmp_path / "ckpt.pt"
    _stub_download(monkeypatch, lambda d: _zip(d, n=6))
    out = weights.fetch_url("https://x/ckpt.pt", dest)
    assert out == dest and weights.artifact_intact(dest)
    assert not list(tmp_path.glob(".*.part")), "staging left behind"


def test_url_truncated_result_raises_and_leaves_no_file(tmp_path, monkeypatch):
    dest = tmp_path / "ckpt.pt"
    _stub_download(monkeypatch, lambda d: _truncate(_zip(d)))
    with pytest.raises(RuntimeError, match="integrity check twice"):
        weights.fetch_url("https://x/ckpt.pt", dest)
    assert not dest.exists()
    assert not list(tmp_path.glob(".*.part"))


def test_url_size_mismatch_is_rejected(tmp_path, monkeypatch):
    """A zip can be a valid archive and still be the wrong file. Content-Length from
    the source of record catches what the structural check cannot, which is what the
    500 GB MSA tarballs rely on entirely."""
    dest = tmp_path / "db.tar.gz"
    url = "https://x/db.tar.gz"
    _stub_download(monkeypatch, lambda d: d.write_bytes(b"z" * 100) and None,
                   sizes={url: 999})
    with pytest.raises(RuntimeError, match="integrity check twice"):
        weights.fetch_url(url, dest, check_archive=False)
    assert not dest.exists()


# --------------------------------------------------------------------------
# Derived directories: the extract-then-discard trap
# --------------------------------------------------------------------------

_EXPECT = ("token_initializer.real_weights.pt", "token_initializer.real_weights.meta.json",
           "diffusion_module.real_weights.pt", "diffusion_module.real_weights.meta.json")
_SPEC = weights.Derived("rfd3/weights", "rfd3", expect=_EXPECT, discard_archive=True)


def _write_outputs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in _EXPECT:
        _zip(out / name) if name.endswith(".pt") else (out / name).write_text('{"n": 1}')


def _stub_produce(monkeypatch, fn):
    monkeypatch.setattr(weights, "_produce",
                        lambda producer, archive, staging, quiet=False: fn(staging))


def test_partial_extraction_is_rebuilt_not_adopted(tmp_path, monkeypatch):
    """The gate used to name one output file: the last and largest one written, so a
    kill during that write left exactly the file it checks, truncated."""
    out = tmp_path / "rfd3" / "weights"
    _write_outputs(out)
    _truncate(out / "diffusion_module.real_weights.pt")
    archive = _zip(tmp_path / "rfd3.ckpt")
    _stub_produce(monkeypatch, _write_outputs)
    result = weights.ensure_derived(archive, _SPEC, root=tmp_path)
    assert all(weights.artifact_intact(result / n) for n in _EXPECT)


def test_archive_is_discarded_only_after_the_output_verifies(tmp_path, monkeypatch):
    """Deleting the checkpoint next to a half-written output is what makes the RFD3
    case permanent rather than merely annoying."""
    archive = _zip(tmp_path / "rfd3.ckpt")
    _stub_produce(monkeypatch, lambda staging: _write_outputs(staging) or
                  _truncate(staging / "diffusion_module.real_weights.pt"))
    with pytest.raises(RuntimeError, match="incomplete"):
        weights.ensure_derived(archive, _SPEC, root=tmp_path)
    assert archive.exists(), "source deleted despite a failed extraction"
    assert not (tmp_path / "rfd3" / "weights").exists()

    _stub_produce(monkeypatch, _write_outputs)
    weights.ensure_derived(archive, _SPEC, root=tmp_path)
    assert not archive.exists(), "source kept after a good extraction"


def test_complete_directory_is_adopted_without_reextracting(tmp_path, monkeypatch):
    """No host may re-download or re-extract what it already has. ~65 GB per host
    times five hosts makes a re-fetch a net loss even when it is technically correct."""
    out = tmp_path / "rfd3" / "weights"
    _write_outputs(out)
    stamps = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    archive = tmp_path / "rfd3.ckpt"        # already discarded by an earlier run
    _stub_produce(monkeypatch, lambda staging: pytest.fail("re-extracted a good output"))
    result = weights.ensure_derived(archive, _SPEC, root=tmp_path)
    assert {p.name: p.stat().st_mtime_ns for p in result.iterdir() if p.is_file()} == stamps


def test_tar_extraction_promotes_the_single_top_level_dir(tmp_path, monkeypatch):
    """mols.tar unpacks one top-level mols/, and the staged tree has to be exactly
    what lands at the destination or the rename puts it one level too deep."""
    src = tmp_path / "src" / "mols"
    src.mkdir(parents=True)
    for i in range(30):
        (src / f"m{i}.pkl").write_bytes(b"m")
    tar = tmp_path / "mols.tar"
    with tarfile.open(tar, "w") as t:
        t.add(src, arcname="mols")
    spec = weights.Derived("mols", "tar", min_entries=30)
    out = weights.ensure_derived(tar, spec, root=tmp_path)
    assert out == tmp_path / "mols"
    assert len(list(out.iterdir())) == 30


def test_completion_marker_lives_outside_the_output(tmp_path, monkeypatch):
    """Writing it inside would change the extracted tree; mols/ is globbed by name."""
    archive = _zip(tmp_path / "rfd3.ckpt")
    _stub_produce(monkeypatch, _write_outputs)
    out = weights.ensure_derived(archive, _SPEC, root=tmp_path)
    assert sorted(p.name for p in out.iterdir()) == sorted(_EXPECT)
    assert (out.parent / f".complete-{out.name}").exists()


# --------------------------------------------------------------------------
# Manual rows are verified, never fetched
# --------------------------------------------------------------------------

def test_openfold3_is_never_downloaded(tmp_path, monkeypatch):
    """No parameter licence is published, so the row is verify-only by design."""
    monkeypatch.setenv("TT_BIO_CACHE", str(tmp_path))
    for var in ("OF3_CKPT", "TT_BIO_OPENFOLD3"):
        monkeypatch.delenv(var, raising=False)
    assert weights.ARTIFACTS["openfold3"].source == "manual"
    with pytest.raises(FileNotFoundError, match="does not download it"):
        weights.fetch("openfold3")


def _only_override(monkeypatch, key: str, path: Path) -> None:
    """Point one row at `path` and clear every other override it accepts.

    Setting the canonical var is not enough: `env_vars` lists `legacy_env` first, so a
    stale legacy override in the ambient environment wins and the row silently resolves
    to whatever that names. The release gate's own env.sh exports `OF3_CKPT`, which is
    how a truncated-checkpoint test came to read a real, intact checkpoint and see no
    error at all. Derived from the registry so a new legacy var can't reopen the hole.
    """
    for var in weights.ARTIFACTS[key].env_vars:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(weights.ARTIFACTS[key].env, str(path))


def test_truncated_manual_checkpoint_is_named_by_tt_bio(tmp_path, monkeypatch):
    """Previously this died inside torch.load with no hint at the cause."""
    ckpt = _truncate(_zip(tmp_path / "of3-p2-155k.pt"))
    _only_override(monkeypatch, "openfold3", ckpt)
    with pytest.raises(RuntimeError, match="truncated or corrupt"):
        weights.fetch("openfold3")


# --------------------------------------------------------------------------
# One cache root moves everything
# --------------------------------------------------------------------------

def test_tt_bio_cache_moves_both_halves(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_BIO_CACHE", str(tmp_path))
    for var in ("HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    assert weights.cache_root() == tmp_path
    assert weights.configure_hf_cache() == tmp_path / "hf"


def test_user_hf_setting_is_never_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_BIO_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "/somewhere/mine")
    assert weights.configure_hf_cache() is None


def test_boltz_cache_keeps_its_historical_reach(monkeypatch, tmp_path):
    """$BOLTZ_CACHE moves the flat half only. Widening it to the hub cache would make
    every host that exports it re-download ~44 GB."""
    monkeypatch.delenv("TT_BIO_CACHE", raising=False)
    monkeypatch.setenv("BOLTZ_CACHE", str(tmp_path))
    for var in ("HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    assert weights.cache_root() == tmp_path
    assert weights.configure_hf_cache() is None


def test_default_layout_is_unchanged(monkeypatch):
    """The registry adopts today's paths as its defaults; anything else invalidates
    every host's cache."""
    for var in ("TT_BIO_CACHE", "BOLTZ_CACHE"):
        monkeypatch.delenv(var, raising=False)
    root = weights.cache_root()
    assert root == Path.home() / ".boltz"
    assert weights.ARTIFACTS["boltz2-conf"].dest() == root / "boltz2_conf.ckpt"
    assert weights.ARTIFACTS["boltz2-aff"].dest() == root / "boltz2_aff.ckpt"
    assert weights.ARTIFACTS["protenix-v2"].dest() == root / "protenix-v2.pt"
    assert weights.ARTIFACTS["mols"].derived_dest() == root / "mols"
    assert weights.ARTIFACTS["boltzgen-diverse"].dest() == root / "boltzgen/boltzgen1_diverse.ckpt"
    assert weights.ARTIFACTS["rfd3"].derived_dest() == root / "rfd3/weights"
    assert weights.ARTIFACTS["rf3"].dest() == root / "rf3/rf3_foundry_01_24_latest_remapped.ckpt"
    assert weights.ARTIFACTS["openfold3"].filename == "of3-p2-155k.pt"
