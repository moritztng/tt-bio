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
import time
import zipfile
from pathlib import Path

import pytest

from tt_bio import weights


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
    missing from it would silently get no prefetch, no status row and no docs entry.

    The tuples are DISCOVERED, not named. Written against the four that existed then, this
    check was blind to a new CLI verb bringing its own tuple: `tt-bio affinity` added
    AFFINITY_MODELS, so nesso1 shipped with no registry row, no `tt-bio weights` entry and
    no docs row while the check built to make exactly that impossible stayed green. Same
    idiom as scripts/perf_regression.py:_assert_full_model_coverage, which got it right.
    """
    from tt_bio import main as _main

    tuples = {n: getattr(_main, n) for n in dir(_main) if n.endswith("_MODELS")}
    assert len(tuples) >= 5, f"expected main.py's --model tuples, found {sorted(tuples)}"
    for model in sorted(set().union(*tuples.values())):
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
# fetch_file: one staged, verified, atomically renamed path for every transport
# --------------------------------------------------------------------------

def _stub_hub(monkeypatch, produce):
    """Replace hf_hub_download with `produce(staging_dir, filename) -> path`."""
    import huggingface_hub

    def fake(repo_id, filename, local_dir, force_download=False, **kw):
        return str(produce(Path(local_dir), filename))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)


HUB = "hf://repo/w.ckpt"


def test_corrupt_cached_file_is_refetched(tmp_path, monkeypatch):
    dest = _truncate(_zip(tmp_path / "w.ckpt"))
    broken = dest.stat().st_size
    _stub_hub(monkeypatch, lambda d, f: _zip(d / f, n=8))
    out = weights.fetch_file((HUB,), dest)
    assert out == dest
    assert weights.artifact_intact(out) and out.stat().st_size != broken


def test_intact_cached_file_is_not_refetched(tmp_path, monkeypatch):
    """The property that makes this safe to deploy on hosts holding ~65 GB of weights."""
    dest = _zip(tmp_path / "w.ckpt")
    before = dest.stat().st_mtime_ns
    _stub_hub(monkeypatch, lambda d, f: pytest.fail("re-downloaded an intact artifact"))
    monkeypatch.setattr(weights, "remote_size",
                        lambda *a, **k: pytest.fail("asked the network about a cached file"))
    assert weights.fetch_file((HUB,), dest) == dest
    assert dest.stat().st_mtime_ns == before


def test_truncated_download_never_reaches_the_final_path(tmp_path, monkeypatch):
    """The core guarantee: a bad download raises and leaves the good copy in place."""
    dest = _zip(tmp_path / "w.ckpt")
    keep = dest.read_bytes()
    _stub_hub(monkeypatch, lambda d, f: _truncate(_zip(d / f)))
    with pytest.raises(weights.DownloadFailed, match="truncated"):
        weights.fetch_file((HUB,), dest, force=True)
    assert dest.read_bytes() == keep
    assert not list(tmp_path.glob(".dl-*")), "staging left behind"


def _stub_download(monkeypatch, produce, sizes=None, fail=()):
    """Stand in for the network. ``fail`` sources deliver nothing, as a host that
    cannot be reached from this machine does."""
    monkeypatch.setattr(weights, "remote_size",
                        lambda url, timeout=15.0, deadline=30.0: (sizes or {}).get(url))

    def download(source, dest, *, quiet=False, attempts=None):
        if source in fail:
            if attempts is not None:
                attempts.append((source, "aria2c", "nothing arrived in 60s", 0))
            return False, False
        produce(dest)
        return True, True

    monkeypatch.setattr(weights, "_download_to", download)


def test_url_download_is_staged_then_renamed(tmp_path, monkeypatch):
    dest = tmp_path / "ckpt.pt"
    _stub_download(monkeypatch, lambda d: _zip(d, n=6))
    out = weights.fetch_file(("https://x/ckpt.pt",), dest)
    assert out == dest and weights.artifact_intact(dest)
    assert not list(tmp_path.glob(".*.part")), "staging left behind"


def test_url_size_mismatch_is_rejected(tmp_path, monkeypatch):
    """A zip can be a valid archive and still be the wrong file. Content-Length from
    the source of record catches what the structural check cannot, which is what the
    500 GB MSA tarballs rely on entirely."""
    dest = tmp_path / "db.tar.gz"
    url = "https://x/db.tar.gz"
    _stub_download(monkeypatch, lambda d: d.write_bytes(b"z" * 100) and None,
                   sizes={url: 999})
    with pytest.raises(weights.DownloadFailed, match="server says 999"):
        weights.fetch_file((url,), dest, check_archive=False)
    assert not dest.exists()


# --------------------------------------------------------------------------
# A source that will not serve: bounded, ordered, and loud
# --------------------------------------------------------------------------

def test_a_source_that_delivers_nothing_falls_over_to_the_next_one(tmp_path, monkeypatch):
    """The customer failure, in one test: the first host never sends a byte.

    It has to cost a retry, not the install. Before this, the only source for the
    Protenix checkpoint was a bucket in Beijing and the download waited on it forever."""
    dest = tmp_path / "ckpt.pt"
    dead, alive = "https://dead/ckpt.pt", "https://alive/ckpt.pt"
    _stub_download(monkeypatch, lambda d: _zip(d, n=6), fail=(dead,))
    assert weights.fetch_file((dead, alive), dest) == dest
    assert weights.artifact_intact(dest)


def test_a_hub_source_that_fails_falls_over_to_the_direct_url(tmp_path, monkeypatch):
    """The two transports are one ordered list, so the fallback crosses between them.

    This is the shape `protenix-v1` ships as: the hub copy first, upstream behind it."""
    import huggingface_hub

    dest = tmp_path / "w.ckpt"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kw: (_ for _ in ()).throw(OSError("hub unreachable")))
    monkeypatch.setattr(weights, "remote_size", lambda *a, **k: None)
    monkeypatch.setattr(weights, "_tool_commands",
                        lambda url, d: [("stub", ["cp", str(_zip(tmp_path / "src.ckpt", n=6)),
                                                  str(d)])])
    assert weights.fetch_file((HUB, "https://upstream/w.ckpt"), dest) == dest
    assert weights.artifact_intact(dest)


def test_when_no_source_serves_the_error_names_every_one_of_them(tmp_path, monkeypatch):
    dest = tmp_path / "ckpt.pt"
    urls = ("https://mirror/ckpt.pt", "https://upstream/ckpt.pt")
    _stub_download(monkeypatch, lambda d: None, fail=urls)
    with pytest.raises(weights.DownloadFailed) as e:
        weights.fetch_file(urls, dest)
    for url in urls:
        assert url in str(e.value)
    assert e.value.bytes_received == 0
    assert not dest.exists() and not list(tmp_path.glob(".*.part"))


def test_a_download_that_stops_moving_is_killed(tmp_path, monkeypatch):
    """The bound that actually holds is ours: watch the bytes on disk, not the tool.

    A tool asked to fetch from a host that accepts the connection and sends nothing
    reports nothing and exits never, so the watchdog is what ends it. Driven here with
    a command that simply sleeps, which is indistinguishable from that case on disk."""
    monkeypatch.setattr(weights, "NO_START_SECONDS", 2)
    t0 = time.monotonic()
    good, reason, got, dead = weights._run_tool(
        "sleep", ["sleep", "300"], tmp_path / "ckpt.pt", total=None, quiet=True)
    assert not good and dead and got == 0
    assert "nothing arrived" in reason
    assert time.monotonic() - t0 < 20, "the watchdog did not fire"


def test_no_download_tool_runs_without_a_timeout(tmp_path):
    """--max-tries=0 is aria2's word for "retry forever", and it cost a customer a day.

    Every tool this path can pick must carry a connect timeout and a give-up rule, so
    the assertion is on the commands themselves rather than on one tool."""
    cmds = weights._tool_commands("https://x/ckpt.pt", tmp_path / "ckpt.pt")
    assert cmds, "no download tool found on this host"
    for name, cmd in cmds:
        flat = " ".join(cmd)
        assert "--max-tries=0" not in flat, name
        assert "timeout" in flat or "--speed-time" in flat, name
        assert any(t in flat for t in ("--max-tries=3", "--retry 3", "--tries=3")), name


def test_a_wrong_file_is_rejected_by_its_hash(tmp_path, monkeypatch):
    """A mirror can serve the wrong object with a perfectly valid archive in it."""
    dest = tmp_path / "ckpt.pt"
    _stub_download(monkeypatch, lambda d: _zip(d, n=6))
    with pytest.raises(weights.DownloadFailed, match="sha256"):
        weights.fetch_file(("https://x/ckpt.pt",), dest, sha256="00" * 32)
    assert not dest.exists()


def test_protenix_v1_prefers_the_hub_and_keeps_upstream():
    """Order is the fix. Upstream stays: it is where the bytes came from."""
    art = weights.ARTIFACTS["protenix-v1"]
    assert art.sources == (f"hf://{art.repo}/{art.filename}", art.url)
    assert art.repo.startswith("moritztng/")
    assert "volces.com" in art.url
    assert art.sha256 and len(art.sha256) == 64


def test_no_row_names_a_bucket_we_no_longer_publish_to():
    """The interim Protenix mirror lived in a GCS bucket that is now deleted. A row
    still pointing at it would be a source that 404s on every install."""
    for art in weights._ROWS:
        for src in art.sources:
            assert "tt-boltz-artifacts" not in src, art.key


def test_every_row_is_fetchable_by_the_one_path_that_exists():
    """One transport list per row, so `fetch` needs no per-row special case: a `file`
    row has at least one source, and the other two kinds are the hub and by hand."""
    for art in weights._ROWS:
        assert art.source in ("file", "hf-repo", "manual"), art.key
        if art.source == "file":
            assert art.sources, art.key
            assert all(weights.source_host(s) for s in art.sources), art.key
        else:
            assert not art.sources, art.key


def test_unavailable_weights_say_what_to_do_next(tmp_path):
    """We cannot log in to her machine, so the exception has to carry the next step."""
    art = weights.ARTIFACTS["protenix-v1"]
    failure = weights.DownloadFailed("model_v0.5.0.pt",
                                     [(s, "aria2c", "nothing arrived in 60s", 0)
                                      for s in art.sources])
    msg = str(weights.WeightsUnavailable(art, failure, tmp_path))
    for expected in (art.repo, art.url, art.sha256, art.env,
                     "tt-bio preflight", "Not one byte arrived"):
        assert expected in msg
    assert "hf://" not in msg, "an hf:// source is not a link a person can open"
    assert str(art.dest(tmp_path)) in msg


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
