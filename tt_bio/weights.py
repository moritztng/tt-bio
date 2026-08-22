"""Every weight and data artifact tt-bio downloads, in one registry, behind one fetch.

Two problems this solves.

**Cache poisoning.** A download killed mid-flight (worker respawn, watchdog, SIGKILL,
a network blip, a half-finished copy between hosts) leaves a truncated multi-GB file.
Gating re-download on ``path.exists()`` alone then treats that file as present and
reuses it forever, surfacing as ``PytorchStreamReader ... failed finding central
directory``. Every fetch here stages into a temporary path, verifies the result, and
only then renames it into place, so the final path never holds an incomplete file. The
same rule covers archives we unpack: the output directory is built under a staging name
and renamed in, and the source archive is only discarded after the output verifies.

**No inventory.** "Every artifact we ship" was spread across ``main.py``, ``worker.py``,
``esmc.py``, ``saprot.py``, ``opendde.py`` and the vendored BoltzGen CLI, so docs, the
release gate, prefetching and disk audits each re-typed their own partial list.
``ARTIFACTS`` is now the single source of truth, the same way ``PREDICT_MODELS`` is for
model names.

Cache layout is unchanged by design: ``$BOLTZ_CACHE`` (default ``~/.boltz``) holds the
flat checkpoints, the Hugging Face hub cache holds the rest. ``$TT_BIO_CACHE`` is the
one knob that relocates both. See ``docs/weights.md``.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache roots
# ---------------------------------------------------------------------------

def cache_root(root: str | Path | None = None) -> Path:
    """Directory holding the flat (non-hub) artifacts.

    An explicit ``root`` (the CLI's ``--cache``) wins, then ``$TT_BIO_CACHE``, then
    ``$BOLTZ_CACHE``, then ``~/.boltz``. The two env vars differ in reach, not in this
    path: ``$TT_BIO_CACHE`` also moves the Hugging Face hub cache (see
    ``configure_hf_cache``), ``$BOLTZ_CACHE`` keeps its historical meaning of "the flat
    half only" so no existing host changes behaviour."""
    if root:
        return Path(root).expanduser()
    env = os.environ.get("TT_BIO_CACHE") or os.environ.get("BOLTZ_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".boltz"


def configure_hf_cache() -> Path | None:
    """Default the Hugging Face hub cache under ``$TT_BIO_CACHE`` when one is set.

    Without this, relocating tt-bio's weights takes two env vars and only one of them
    is documented, which is how a shared box ends up with 44 GB of hub cache on the
    wrong filesystem. A user who set ``HF_HOME`` or ``HF_HUB_CACHE`` themselves is
    left alone. Returns the value it set, or None.

    Must run before ``huggingface_hub`` is imported (it reads the env at import time),
    which is why ``tt_bio/__init__.py`` calls it."""
    root = os.environ.get("TT_BIO_CACHE")
    if not root or os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME"):
        return None
    hub = Path(root).expanduser() / "hf"
    os.environ["HF_HUB_CACHE"] = str(hub)
    return hub


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Derived:
    """A directory produced from an artifact (archive extraction, weight split).

    ``expect``/``min_entries`` are how we tell a complete output from the wreckage of
    an interrupted one. They also let an already-populated directory be adopted
    without redoing the work, so upgrading tt-bio never re-extracts or re-downloads
    what a host already has."""

    subdir: str                          # cache-relative output directory
    producer: str                        # "tar" | "rfd3"
    expect: tuple[str, ...] = ()         # files that must exist and verify inside
    min_entries: int = 0                 # or: at least this many entries
    discard_archive: bool = False        # unlink the source once the output verifies


@dataclass(frozen=True)
class Artifact:
    """One downloadable thing, and everything any consumer needs to know about it."""

    key: str
    models: tuple[str, ...]              # CLI model names that load it
    source: str                          # "hf-file" | "hf-repo" | "url" | "manual"
    licence: str
    repo: str | None = None              # HF repo id (hf-file, hf-repo)
    filename: str | None = None          # path within the repo (hf-file, hf-repo)
    url: str | None = None               # direct download (url)
    subdir: str = ""                     # cache-relative dir for flat files
    approx_bytes: int = 0                # measured, for the size column and prefetch
    derived: Derived | None = None
    legacy_env: tuple[str, ...] = ()     # pre-registry overrides, still honoured
    note: str = ""

    @property
    def env(self) -> str:
        """Canonical override env var, derived from the key so it can never drift."""
        return "TT_BIO_" + self.key.upper().replace("-", "_").replace(".", "_")

    @property
    def env_vars(self) -> tuple[str, ...]:
        """Every override accepted, most specific first."""
        return (*self.legacy_env, self.env)

    def dest(self, root: str | Path | None = None) -> Path:
        """Where a flat artifact lives. Meaningless for ``hf-repo``."""
        return cache_root(root) / self.subdir / (self.filename or Path(self.url or "").name)

    def derived_dest(self, root: str | Path | None = None) -> Path | None:
        return cache_root(root) / self.derived.subdir if self.derived else None


_GB = 1 << 30
_MB = 1 << 20

BOLTZ2_REPO = "moritztng/boltz-2"
PROTENIX_REPO = "TMF001/protenix-v2-weights"
BOLTZGEN_REPO = "moritztng/boltzgen"
OPENDDE_REPO = "aurekaresearch/OpenDDE"
IPD_BASE = "https://files.ipd.uw.edu/pub"

# Sizes come from the source of record (the HF repo's file metadata, or a populated
# host for the IPD downloads), not from an estimate. They drive the size column, the
# prefetch total and the disk audit; nothing depends on them being exact. Reading them
# off local disk would be wrong: pc's cached BoltzGen affinity checkpoint is truncated,
# so "measured locally" would have recorded the corruption as the expected size.
_ROWS: tuple[Artifact, ...] = (
    # -- Boltz-2 + the shared CCD molecule library ------------------------------
    Artifact("boltz2-conf", ("boltz2",), "hf-file", "MIT",
             repo=BOLTZ2_REPO, filename="boltz2_conf.ckpt", approx_bytes=2286561469),
    Artifact("boltz2-aff", ("boltz2",), "hf-file", "MIT",
             repo=BOLTZ2_REPO, filename="boltz2_aff.ckpt", approx_bytes=2062139170,
             note="affinity head; only read for ligand affinity"),
    Artifact("mols", ("boltz2", "protenix-v2"), "hf-file", "MIT",
             repo=BOLTZ2_REPO, filename="mols.tar", approx_bytes=1855662080,
             derived=Derived("mols", "tar", min_entries=45227),
             note="CCD molecule library, extracted to <cache>/mols"),

    # -- Protenix-v2 ------------------------------------------------------------
    Artifact("protenix-v2", ("protenix-v2",), "hf-file", "Apache-2.0",
             repo=PROTENIX_REPO, filename="protenix-v2.pt", approx_bytes=1859785497,
             legacy_env=("PROTENIX_CKPT",)),

    # -- ESMFold2 / ESMC / SaProt: whole HF repos, read from the hub cache ------
    Artifact("esmfold2", ("esmfold2",), "hf-repo", "non-commercial (EvolutionaryScale)",
             repo="biohub/ESMFold2", approx_bytes=1352914698),
    Artifact("esmfold2-fast", ("esmfold2-fast",), "hf-repo", "non-commercial (EvolutionaryScale)",
             repo="biohub/ESMFold2-Fast", approx_bytes=751619276),
    Artifact("esmc-300m", ("esmc-300m",), "hf-repo", "non-commercial (EvolutionaryScale)",
             repo="biohub/esmc-300m-2024-12", filename="data/weights/esmc_300m_2024_12_v0.pth",
             approx_bytes=1331439861),
    Artifact("esmc-600m", ("esmc-600m",), "hf-repo", "non-commercial (EvolutionaryScale)",
             repo="biohub/esmc-600m-2024-12", filename="data/weights/esmc_600m_2024_12_v0.pth",
             approx_bytes=2297556992),
    Artifact("esmc-6b", ("esmc-6b",), "hf-repo", "non-commercial (EvolutionaryScale)",
             repo="biohub/ESMC-6B", approx_bytes=25405672653),
    Artifact("saprot-35m", ("saprot-35m",), "hf-repo", "MIT",
             repo="westlake-repl/SaProt_35M_AF2", approx_bytes=408021893),
    Artifact("saprot-650m", ("saprot-650m",), "hf-repo", "MIT",
             repo="westlake-repl/SaProt_650M_AF2", approx_bytes=7816264089),
    # The 1.3B is the one row no host has cached, so its size is scaled from the 650M's
    # measured 7.28 GiB rather than measured. Nothing depends on it being exact.
    Artifact("saprot-1.3b", ("saprot-1.3b",), "hf-repo", "MIT",
             repo="westlake-repl/SaProt_1.3B_AF2", approx_bytes=15000000000),

    # -- OpenDDE --------------------------------------------------------------
    Artifact("opendde", ("opendde",), "hf-repo", "see repo card (Aureka Research)",
             repo=OPENDDE_REPO, filename="opendde.pt", approx_bytes=2625249069,
             legacy_env=("OPENDDE_CKPT",)),
    Artifact("opendde-abag", ("opendde-abag",), "hf-repo", "see repo card (Aureka Research)",
             repo=OPENDDE_REPO, filename="opendde_abag.pt", approx_bytes=2625271509,
             legacy_env=("OPENDDE_CKPT",)),

    # -- BoltzGen: six flat files under <cache>/boltzgen -----------------------
    Artifact("boltzgen-diverse", ("boltzgen",), "hf-file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltzgen1_diverse.ckpt", subdir="boltzgen",
             approx_bytes=1930847192),
    Artifact("boltzgen-adherence", ("boltzgen",), "hf-file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltzgen1_adherence.ckpt", subdir="boltzgen",
             approx_bytes=1930858014),
    Artifact("boltzgen-ifold", ("boltzgen",), "hf-file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltzgen1_ifold.ckpt", subdir="boltzgen",
             approx_bytes=12582656),
    Artifact("boltzgen-folding", ("boltzgen",), "hf-file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltz2_conf_final.ckpt", subdir="boltzgen",
             approx_bytes=2087255089),
    Artifact("boltzgen-affinity", ("boltzgen",), "hf-file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltz2_aff.ckpt", subdir="boltzgen",
             approx_bytes=2061914091),
    Artifact("boltzgen-mols", ("boltzgen",), "hf-file", "MIT",
             repo=BOLTZGEN_REPO, filename="mols.zip", subdir="boltzgen",
             approx_bytes=391401102, note="read as a zip, not extracted"),

    # -- IPD direct downloads --------------------------------------------------
    Artifact("rf3", ("rf3",), "url", "see files.ipd.uw.edu (Institute for Protein Design)",
             url=f"{IPD_BASE}/rf3/rf3_foundry_01_24_latest_remapped.ckpt", subdir="rf3",
             approx_bytes=3038876446, legacy_env=("RF3_CKPT",)),
    Artifact("rfd3", ("rfd3",), "url", "see files.ipd.uw.edu (Institute for Protein Design)",
             url=f"{IPD_BASE}/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt", subdir="rfd3",
             approx_bytes=2690316669,
             derived=Derived("rfd3/weights", "rfd3", discard_archive=True, expect=(
                 "token_initializer.real_weights.pt", "token_initializer.real_weights.meta.json",
                 "diffusion_module.real_weights.pt", "diffusion_module.real_weights.meta.json")),
             note="only the extracted TokenInitializer/DiffusionModule weights are kept"),

    # -- OpenFold3: no auto-download, on purpose -------------------------------
    Artifact("openfold3", ("openfold3",), "manual", "no parameter licence published",
             filename="of3-p2-155k.pt", approx_bytes=2287928196, legacy_env=("OF3_CKPT",),
             note="fetch from the OpenFold consortium yourself; see README"),
    Artifact("openbind", ("openbind",), "manual", "no parameter licence published",
             filename="of3-ob-2025-06-30-174k.pt", approx_bytes=2287872989,
             note="OpenBind-0, upstream openfold-3 v0.5.0. Ungated at "
                  "https://openfold3-data.s3.amazonaws.com/openfold3-parameters/"
                  "of3-ob-2025-06-30-174k.pt, but treated as manual like the preview2 "
                  "row above: the code is Apache-2.0 and the parameters carry no "
                  "separate licence we can point at. See docs/weights.md"),
)

ARTIFACTS: dict[str, Artifact] = {a.key: a for a in _ROWS}

# model -> the artifacts it needs, derived so a new row is picked up automatically.
MODEL_ARTIFACTS: dict[str, tuple[str, ...]] = {}
for _a in _ROWS:
    for _m in _a.models:
        MODEL_ARTIFACTS[_m] = (*MODEL_ARTIFACTS.get(_m, ()), _a.key)


def artifacts_for(*models: str) -> tuple[Artifact, ...]:
    """Registry rows needed by the named models (all rows when none are named)."""
    if not models:
        return _ROWS
    keys: list[str] = []
    for m in models:
        if m not in MODEL_ARTIFACTS:
            raise KeyError(f"unknown model {m!r}; known: {', '.join(sorted(MODEL_ARTIFACTS))}")
        keys.extend(k for k in MODEL_ARTIFACTS[m] if k not in keys)
    return tuple(ARTIFACTS[k] for k in keys)


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def artifact_intact(path: Path, name: str | None = None) -> bool:
    """Is a cached artifact actually complete?

    Our checkpoints (.ckpt/.pt/.pth) and the molecule bundle (.zip) are PyTorch/zip
    archives, so reading the zip central directory is a cheap and decisive completeness
    check: a truncated file has no valid central directory, which is exactly the
    ``PytorchStreamReader ... failed finding central directory`` failure. A tar is
    checked by reading its member stream header. Anything else: require a non-empty
    file. ``name`` judges a staging file by its eventual name."""
    try:
        suffix = Path(name or path.name).suffix.lower()
        if suffix in (".ckpt", ".pt", ".pth", ".zip"):
            import zipfile
            return zipfile.is_zipfile(path)
        if suffix in (".tar", ".tgz", ".gz", ".bz2", ".xz"):
            import tarfile
            # Reading only the first header would miss the common case: a tar truncated
            # at the end still has a valid first member. Walking to the last member is
            # the decisive check (6.5 s for mols.tar's 45 228 members, paid only when we
            # actually download, since the completion marker covers the steady state).
            with tarfile.open(path) as tar:
                seen = False
                while tar.next() is not None:
                    seen = True
                return seen
        if suffix == ".json":
            import json
            with open(path) as fh:
                json.load(fh)
            return True
        return path.stat().st_size > 0
    except Exception:
        return False


def sweep_stale_staging(cache: Path, max_age_s: float = 3600.0) -> None:
    """Remove orphaned ``.dl-*`` / ``.stage-*`` entries left by a hard-killed run.

    Staging is cleaned in a ``finally``, which a SIGKILL skips, so it lingers forever.
    Harmless (each run picks a unique name) but unbounded. Anything from a prior
    process is safe to delete; the mtime gate keeps us off an in-flight download, and
    multi-device workers on one host start within seconds of each other so an hour of
    slack is generous. Resumable ``.part`` files are deliberately left alone."""
    if not cache.is_dir():
        return
    cutoff = time.time() - max_age_s
    for pattern in (".dl-*", "*.stage-*"):
        for entry in cache.glob(pattern):
            try:
                if entry.stat().st_mtime >= cutoff:
                    continue
                shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
            except OSError:
                pass


def _echo(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# The one fetch path: stage -> verify -> atomic rename
# ---------------------------------------------------------------------------

def fetch_hf_file(repo_id: str, filename: str, dest_dir: Path, *,
                  force: bool = False, quiet: bool = False) -> Path:
    """Fetch one file from a HF repo to ``dest_dir/<basename>``, atomically.

    Re-fetches when the local copy is missing *or* corrupt. Trusting mere existence is
    what lets a truncated file poison the cache permanently."""
    import tempfile

    dest_dir = Path(dest_dir)
    result = dest_dir / Path(filename).name
    if not force and result.exists() and artifact_intact(result):
        return result
    if result.exists() and not force:
        _echo(f"Cached {result.name} is incomplete/corrupt, re-downloading", quiet)
    dest_dir.mkdir(parents=True, exist_ok=True)
    sweep_stale_staging(dest_dir)
    _echo(f"Downloading {filename} from {repo_id}", quiet)

    from huggingface_hub import hf_hub_download
    staging = Path(tempfile.mkdtemp(dir=str(dest_dir), prefix=".dl-"))
    try:
        tmp = Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                   local_dir=str(staging), force_download=True))
        if not artifact_intact(tmp, result.name):
            raise RuntimeError(
                f"downloaded {filename} failed its integrity check (truncated/corrupt "
                f"archive), refusing to cache it; please retry")
        os.replace(tmp, result)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return result


def remote_size(url: str, timeout: float = 15.0) -> int | None:
    """Content-Length for ``url``, or None if the server will not say.

    A byte-exact size from the source of record is the strongest completeness check
    there is, and unlike the archive-structure check it works on any file type. That
    matters for the MSA database tarballs, which are far too large to scan."""
    import urllib.request
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            n = resp.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def fetch_url(url: str, dest: Path, *, force: bool = False, quiet: bool = False,
              check_archive: bool = True) -> Path:
    """Download ``url`` to ``dest``, atomically, resuming across runs.

    Staging is a stable ``.<name>.part`` next to the destination so an interrupted
    multi-GB download resumes instead of restarting, while the destination itself only
    ever holds a verified file. A ``.part`` that fails verification is discarded and
    retried once from scratch, since a mid-file corruption can never be fixed by
    resuming. ``check_archive=False`` skips the structural check for a file too large
    to scan (the MSA databases), leaving the byte-count check to carry it."""
    dest = Path(dest)
    expect = None

    def ok(path: Path) -> bool:
        if expect is not None and path.stat().st_size != expect:
            return False
        return artifact_intact(path, dest.name) if check_archive else path.stat().st_size > 0

    if not force and dest.exists():
        expect = remote_size(url)
        if ok(dest):
            return dest
        _echo(f"Cached {dest.name} is incomplete/corrupt, re-downloading", quiet)
    elif not force:
        expect = remote_size(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f".{dest.name}.part")

    for attempt in (1, 2):
        _download_to(url, part, quiet=quiet)
        if ok(part):
            os.replace(part, dest)
            return dest
        part.unlink(missing_ok=True)
        if attempt == 1:
            _echo(f"{dest.name} failed its integrity check, retrying from scratch", quiet)
    raise RuntimeError(
        f"{dest.name} failed its integrity check twice (truncated/corrupt), refusing to "
        f"cache it; please retry or download {url} by hand")


def _download_to(url: str, dest: Path, *, max_retries: int = 5, quiet: bool = False) -> None:
    """Fetch a large file with tool fallback and resume. Never called with a final path."""
    import subprocess
    import urllib.request

    _echo(f"  Downloading {dest.name} ...", quiet)
    tools = []
    if shutil.which("aria2c"):
        tools.append(("aria2c", ["aria2c", "--max-connection-per-server=8", "--split=8",
                                 "--continue=true", "--auto-file-renaming=false",
                                 "--retry-wait=5", "--max-tries=0",
                                 "-o", dest.name, "-d", str(dest.parent), url]))
    if shutil.which("curl"):
        tools.append(("curl", ["curl", "-L", "--retry", "10", "--retry-delay", "5",
                               "-C", "-", "--progress-bar", "-o", str(dest), url]))
    if shutil.which("wget"):
        tools.append(("wget", ["wget", "-c", "--tries=10", "--wait=5", "-O", str(dest), url]))
    if not tools:
        _echo("    (no aria2c/curl/wget, using Python urllib, may be slow)", quiet)
        urllib.request.urlretrieve(url, dest)
        return
    for attempt in range(1, max_retries + 1):
        for name, cmd in tools:
            try:
                subprocess.run(cmd, check=True, capture_output=quiet)
                return
            except subprocess.CalledProcessError:
                _echo(f"    {name} failed (attempt {attempt}/{max_retries})", quiet)
        time.sleep(5)
    raise RuntimeError(f"could not download {url} after {max_retries} attempts")


def fetch_hf_repo(repo_id: str, *, filename: str | None = None, force: bool = False,
                  quiet: bool = False) -> Path:
    """Snapshot a whole HF repo into the hub cache and return the snapshot dir.

    The hub cache is already written blob-at-a-time through ``.incomplete`` staging, so
    the destination cannot hold a partial blob. What it does not do is notice a blob
    that went bad afterwards (a half-finished copy of the cache between hosts, a full
    disk during the final link), so when ``filename`` names the weight file we verify
    it and re-snapshot with ``force_download`` if it fails."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(repo_id, force_download=force))
    if filename:
        target = snap / filename
        if not target.exists() or not artifact_intact(target):
            _echo(f"Cached {repo_id}:{filename} is incomplete/corrupt, re-downloading", quiet)
            snap = Path(snapshot_download(repo_id, force_download=True))
    return snap


# ---------------------------------------------------------------------------
# Derived directories: extract into staging, verify, rename the directory in
# ---------------------------------------------------------------------------

def _marker(out_dir: Path) -> Path:
    """Completion marker, kept beside the output directory rather than inside it so
    the extracted tree stays byte-identical to a pristine extraction."""
    return out_dir.parent / f".complete-{out_dir.name}"


def _derived_ok(out_dir: Path, spec: Derived) -> bool:
    if not out_dir.is_dir():
        return False
    if spec.expect:
        return all(artifact_intact(out_dir / n) for n in spec.expect)
    if spec.min_entries:
        n = 0
        for n, _ in enumerate(out_dir.iterdir(), 1):
            if n >= spec.min_entries:
                return True
        return n >= spec.min_entries
    return any(out_dir.iterdir())


def ensure_derived(archive: Path, spec: Derived, *, root: str | Path | None = None,
                   force: bool = False, quiet: bool = False) -> Path:
    """Produce ``spec.subdir`` from ``archive``, atomically.

    A directory that already exists is adopted when its contents verify (so no host
    ever redoes work it already has) and rebuilt when they do not. The rebuild happens
    under a staging name and is renamed in, so an interrupted extraction cannot leave
    a half-populated directory that passes an existence check. ``discard_archive``
    only fires after the finished output is in place, which is the difference between
    "re-download 2.5 GB" and "permanently poisoned with no path back"."""
    out_dir = cache_root(root) / spec.subdir
    marker = _marker(out_dir)
    if not force and marker.exists() and out_dir.is_dir():
        return out_dir
    if not force and _derived_ok(out_dir, spec):
        marker.write_text("ok\n")            # adopt what this host already has
        return out_dir
    if out_dir.exists():
        _echo(f"{out_dir} is incomplete, rebuilding it", quiet)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    sweep_stale_staging(out_dir.parent)
    staging = out_dir.with_name(f"{out_dir.name}.stage-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        _produce(spec.producer, archive, staging, quiet=quiet)
        if not _derived_ok(staging, spec):
            raise RuntimeError(
                f"extracting {archive.name} produced an incomplete {spec.subdir}, "
                f"refusing to cache it; please retry")
        marker.unlink(missing_ok=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.replace(staging, out_dir)
        marker.write_text("ok\n")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if spec.discard_archive:
        archive.unlink(missing_ok=True)      # only the derived output is read again
    return out_dir


def _produce(producer: str, archive: Path, staging: Path, *, quiet: bool = False) -> None:
    if producer == "tar":
        import tarfile
        _echo(f"Extracting {archive.name}", quiet)
        holding = staging.with_name(staging.name + ".tar")
        shutil.rmtree(holding, ignore_errors=True)
        holding.mkdir(parents=True)
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(holding)
            inner = [p for p in holding.iterdir()]
            # mols.tar unpacks a single top-level `mols/`; promote it so the staged
            # tree is exactly what lands at the destination.
            src = inner[0] if len(inner) == 1 and inner[0].is_dir() else holding
            os.replace(src, staging)
        finally:
            shutil.rmtree(holding, ignore_errors=True)
    elif producer == "rfd3":
        from tt_bio.rfd3.design import extract_rfd3_weights
        _echo("Extracting RFD3 weights", quiet)
        extract_rfd3_weights(archive, staging)
    else:
        raise ValueError(f"unknown producer {producer!r}")


# ---------------------------------------------------------------------------
# Registry-level API
# ---------------------------------------------------------------------------

def _override(art: Artifact) -> Path | None:
    for var in art.env_vars:
        val = os.environ.get(var)
        if val:
            return Path(val).expanduser()
    return None


def resolve(key: str, root: str | Path | None = None) -> Path | None:
    """The path a model actually loads, without fetching. None for an uncached repo.

    For a row with a ``derived`` output that is the output directory, matching what
    ``fetch`` returns, so ``tt-bio weights`` and a fold never disagree about what is
    in use. An override is returned as-is."""
    art = ARTIFACTS[key]
    if (p := _override(art)) is not None:
        return p
    if art.derived:
        return art.derived_dest(root)
    if art.source == "hf-repo":
        from huggingface_hub import try_to_load_from_cache
        if art.filename:
            hit = try_to_load_from_cache(art.repo, art.filename)
            return Path(hit) if isinstance(hit, str) else None
        return _snapshot_dir(art.repo)
    if art.source == "manual":
        return cache_root(root) / art.filename
    return art.dest(root)


def _snapshot_dir(repo_id: str) -> Path | None:
    """Cached snapshot directory for a repo, or None. Avoids a network call."""
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id:
                revs = [r for r in repo.revisions if r.refs] or list(repo.revisions)
                if revs:
                    return Path(revs[0].snapshot_path)
    except Exception:
        pass
    return None


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f), follow_symlinks=True).st_size
            except OSError:
                pass
    return total


@dataclass
class Status:
    """What ``tt-bio weights`` prints for one row."""
    artifact: Artifact
    state: str                # "present" | "missing" | "corrupt" | "partial"
    path: Path | None
    on_disk: int = 0
    override: str | None = None
    extra: str = ""


def status(key: str, root: str | Path | None = None) -> Status:
    """Present/missing/corrupt for one row, judged by the same integrity check the fetch
    path uses, so the table can never disagree with a fold.

    A derived row is judged on its output first. The archive being gone is normal for
    RFD3 (it is deleted after extraction) and harmless for mols once the library is
    unpacked, so a complete output is "present" either way."""
    art = ARTIFACTS[key]
    override = next((v for v in art.env_vars if os.environ.get(v)), None)
    path = resolve(key, root)

    if art.derived:
        out = art.derived_dest(root)
        archive = art.dest(root)
        extra = "" if archive.exists() else "archive not kept"
        if _marker(out).exists() and out.is_dir() or _derived_ok(out, art.derived):
            size = _tree_bytes(out) + (_tree_bytes(archive) if archive.exists() else 0)
            return Status(art, "present", out, size, override, extra)
        if out.exists():
            return Status(art, "partial", out, _tree_bytes(out), override,
                          f"{art.derived.subdir}/ incomplete")
        if not archive.exists():
            return Status(art, "missing", out, 0, override)
        if not artifact_intact(archive):
            return Status(art, "corrupt", archive, _tree_bytes(archive), override,
                          "archive failed integrity check")
        return Status(art, "missing", out, _tree_bytes(archive), override,
                      f"{art.derived.subdir}/ not built")

    if path is None or not path.exists():
        return Status(art, "missing", path, override=override)
    size = _tree_bytes(path)
    if path.is_file() and not artifact_intact(path):
        return Status(art, "corrupt", path, size, override, "failed integrity check")
    return Status(art, "present", path, size, override)


def fetch(key: str, *, root: str | Path | None = None, force: bool = False,
          quiet: bool = False) -> Path:
    """Make one artifact available and return the path a model should load.

    Honours the row's env overrides, verifies whatever is already there, and only
    downloads what is missing or broken. For a row with a ``derived`` output the
    returned path is that output, since it is what the model reads."""
    art = ARTIFACTS[key]
    if (p := _override(art)) is not None:
        if not p.exists():
            raise FileNotFoundError(
                f"${next(v for v in art.env_vars if os.environ.get(v))} points at {p}, "
                f"which does not exist")
        if p.is_file() and not artifact_intact(p):
            raise RuntimeError(
                f"{p} (from ${next(v for v in art.env_vars if os.environ.get(v))}) is "
                f"truncated or corrupt: it is not a readable archive")
        return p

    if art.source == "manual":
        dest = cache_root(root) / art.filename
        if not dest.exists():
            raise FileNotFoundError(
                f"{art.key} checkpoint not found. tt-bio does not download it ({art.licence}). "
                f"Set {' or '.join('$' + v for v in art.env_vars)} to your copy, or place "
                f"it at {dest}.")
        if not artifact_intact(dest):
            raise RuntimeError(
                f"{dest} is truncated or corrupt: it is not a readable archive. Re-copy it "
                f"or point ${art.env} at a good copy.")
        return dest

    if art.source == "hf-repo":
        snap = fetch_hf_repo(art.repo, filename=art.filename, force=force, quiet=quiet)
        return snap / art.filename if art.filename else snap

    if art.source == "hf-file":
        path = fetch_hf_file(art.repo, art.filename, cache_root(root) / art.subdir,
                             force=force, quiet=quiet)
    elif art.source == "url":
        path = art.dest(root)
        if art.derived and art.derived.discard_archive:
            # The archive is deleted after extraction, so its absence is normal. Only
            # fetch it when the derived output is not already good.
            out = cache_root(root) / art.derived.subdir
            if not force and (_marker(out).exists() and out.is_dir() or _derived_ok(out, art.derived)):
                return ensure_derived(path, art.derived, root=root, quiet=quiet)
            _echo(f"Downloading {art.key} checkpoint "
                  f"(~{art.approx_bytes / _GB:.1f} GiB, {art.url.split('/')[2]})", quiet)
        path = fetch_url(art.url, path, force=force, quiet=quiet)
    else:
        raise ValueError(f"unknown source {art.source!r} for {key}")

    if art.derived:
        return ensure_derived(path, art.derived, root=root, force=force, quiet=quiet)
    return path


def fetch_models(*models: str, root: str | Path | None = None,
                 quiet: bool = False) -> dict[str, Path]:
    """Prefetch every artifact the named models need (all models when none named).
    Skips ``manual`` rows, which have nothing to fetch."""
    out: dict[str, Path] = {}
    for art in artifacts_for(*models):
        if art.source == "manual":
            continue
        out[art.key] = fetch(art.key, root=root, quiet=quiet)
    return out


# ---------------------------------------------------------------------------
# Disk audit: what is reclaimable, and what is not ours to touch
# ---------------------------------------------------------------------------

def artifact_paths(key: str, root: str | Path | None = None) -> list[Path]:
    """Everything on disk that belongs to one row: the archive, its derived output and
    the completion marker. Hub-cache rows return their snapshot dir, which is shared
    with the hub's own blob store, so pruning those goes through ``delete_revisions``
    rather than a plain unlink."""
    art = ARTIFACTS[key]
    out: list[Path] = []
    if art.source in ("hf-file", "url", "manual"):
        out.append(art.dest(root))
    if art.derived:
        d = art.derived_dest(root)
        out += [d, _marker(d)]
    return [p for p in out if p.exists()]


def superseded_revisions() -> tuple[list[str], int]:
    """HF hub revisions with no ref, in repos that still have a live one, and the bytes
    they free. Reclaimable size comes from ``delete_revisions``, which refcounts blobs:
    revisions share blobs, so summing per-revision sizes double counts.

    Repos with zero refs are left alone entirely: with nothing live there is no
    "superseded", only a cache we do not understand."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
    except Exception:
        return [], 0
    dead: list[str] = []
    for repo in info.repos:
        if not any(r.refs for r in repo.revisions):
            continue
        dead += [r.commit_hash for r in repo.revisions if not r.refs]
    if not dead:
        return [], 0
    return dead, info.delete_revisions(*dead).expected_freed_size


def delete_revisions(commit_hashes: list[str]) -> int:
    """Delete hub revisions by commit hash; returns the bytes freed."""
    from huggingface_hub import scan_cache_dir

    strategy = scan_cache_dir().delete_revisions(*commit_hashes)
    freed = strategy.expected_freed_size
    strategy.execute()
    return freed


def stale_staging(root: str | Path | None = None) -> list[tuple[Path, int]]:
    """Leftover staging entries from hard-killed runs, anywhere under the cache root."""
    base = cache_root(root)
    out: list[tuple[Path, int]] = []
    if not base.is_dir():
        return out
    for pattern in ("**/.dl-*", "**/*.stage-*", "**/.*.part"):
        for entry in base.glob(pattern):
            out.append((entry, _tree_bytes(entry)))
    return sorted(set(out))


def unmanaged(root: str | Path | None = None) -> list[tuple[Path, int]]:
    """Files and directories under the cache root that no registry row claims.

    Reported, never deleted: the cache is shared (tt-atom's own weights live in the
    same hub cache) and the MSA databases, MSA outputs and template structures all sit
    here legitimately. Naming them with their size is enough for a human to decide."""
    base = cache_root(root)
    if not base.is_dir():
        return []
    claimed = {p.resolve() for k in ARTIFACTS for p in artifact_paths(k, root)}
    claimed |= {(base / a.subdir).resolve() for a in _ROWS if a.subdir}
    keep = {"msa", "msa_db", "msa_server_cache", "demo_msa", "of3_template_structures", ".cache"}
    out = []
    for entry in base.iterdir():
        if entry.name in keep or entry.name.startswith(".") or entry.resolve() in claimed:
            continue
        out.append((entry, _tree_bytes(entry)))
    return sorted(out, key=lambda t: -t[1])


def unmanaged_repos() -> list[tuple[str, int]]:
    """Hub-cache repos no registry row names, largest first. Reported, never deleted:
    tt-atom's weights (facebook/UMA, lab-cosmo/upet) share this cache."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
    except Exception:
        return []
    ours = {a.repo for a in _ROWS if a.repo}
    return sorted(((r.repo_id, r.size_on_disk) for r in info.repos if r.repo_id not in ours),
                  key=lambda t: -t[1])
