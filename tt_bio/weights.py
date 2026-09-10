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
from contextlib import contextmanager
from dataclasses import dataclass
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
    source: str                          # "file" | "hf-repo" | "manual"
    licence: str
    repo: str | None = None              # HF repo id
    revision: str | None = None          # HF tag/commit to pin; None means the repo default
    filename: str | None = None          # path within the repo, or the flat file's name
    url: str | None = None               # direct download, tried after the hub
    sha256: str | None = None            # of the finished file, when we have verified one
    subdir: str = ""                     # cache-relative dir for flat files
    approx_bytes: int = 0                # measured, for the size column and prefetch
    derived: Derived | None = None
    legacy_env: tuple[str, ...] = ()     # pre-registry overrides, still honoured
    note: str = ""

    @property
    def sources(self) -> tuple[str, ...]:
        """Every place this file can be fetched from, best first.

        A ``file`` row may name both a Hugging Face copy and a direct URL, and a row
        that names both is why this is a list rather than a field: a checkpoint served
        from one origin is a checkpoint that hangs for anyone who cannot route to that
        origin. That cost a customer a day on ``protenix-v1``, whose only upstream is a
        single bucket in Beijing. The hub copy leads because huggingface.co answers from
        everywhere and is what the other twelve rows already use; upstream stays as the
        fallback rather than the only door.

        ``hf://<repo>/<file>`` is fetched by the hub client, anything else by the
        download tools. One list of strings, so the fetch path walks it without a
        second concept for "our copy" and an error can name what it tried."""
        out = []
        if self.source == "file" and self.repo and self.filename:
            out.append(f"hf://{self.repo}/{self.filename}")
        if self.url:
            out.append(self.url)
        return tuple(out)

    @property
    def probe_urls(self) -> tuple[str, ...]:
        """URLs to test reachability against, one per host this row depends on.

        A hub source is fetched by the hub client rather than by a URL of ours, so it
        names the API endpoint for its repo: what a diagnosis needs is "can this machine
        talk to that host", and the answer is per host, not per file."""
        if self.source == "hf-repo" and self.repo:
            return (_hf_api_url(self.repo),)
        return tuple(_hf_api_url(hf[0]) if (hf := split_hf(s)) else s for s in self.sources)

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
NESSO_REPO = "recursionpharma/nesso"
NESSO_REVISION = "v1.0.0"      # not `main`: main carries only a config.json
IPD_BASE = "https://files.ipd.uw.edu/pub"
PXDESIGN_BASE = "https://pxdesign.tos-cn-beijing.volces.com"
PROTENIX_V1_REPO = "moritztng/protenix-v0.5.0"

_HF_SCHEME = "hf://"


def split_hf(source: str) -> tuple[str, str] | None:
    """``(repo, filename)`` for an ``hf://repo/file`` source, None for a plain URL."""
    if not source.startswith(_HF_SCHEME):
        return None
    repo, _, filename = source[len(_HF_SCHEME):].rpartition("/")
    return repo, filename


def _hf_api_url(repo: str) -> str:
    return f"https://huggingface.co/api/models/{repo}"


def source_host(source: str) -> str:
    """The host a source is fetched from, for a one-line "where is this coming from"."""
    return "huggingface.co" if split_hf(source) else source.split("/")[2]


def source_url(source: str) -> str:
    """The source as a link a person can open.

    The hub client takes a repo and a filename, but someone told to fetch a checkpoint
    by hand needs a URL, so an error message shows them this instead of ``hf://``."""
    if (hf := split_hf(source)) is not None:
        return f"https://huggingface.co/{hf[0]}/resolve/main/{hf[1]}"
    return source

# Sizes come from the source of record (the HF repo's file metadata, or a populated
# host for the IPD downloads), not from an estimate. They drive the size column, the
# prefetch total and the disk audit; nothing depends on them being exact. Reading them
# off local disk would be wrong: pc's cached BoltzGen affinity checkpoint is truncated,
# so "measured locally" would have recorded the corruption as the expected size.
_ROWS: tuple[Artifact, ...] = (
    # -- Boltz-2 + the shared CCD molecule library ------------------------------
    Artifact("boltz2-conf", ("boltz2",), "file", "MIT",
             repo=BOLTZ2_REPO, filename="boltz2_conf.ckpt", approx_bytes=2286561469),
    Artifact("boltz2-aff", ("boltz2",), "file", "MIT",
             repo=BOLTZ2_REPO, filename="boltz2_aff.ckpt", approx_bytes=2062139170,
             note="affinity head; only read for ligand affinity"),
    Artifact("mols", ("boltz2", "protenix-v1", "protenix-v2"), "file", "MIT",
             repo=BOLTZ2_REPO, filename="mols.tar", approx_bytes=1855662080,
             derived=Derived("mols", "tar", min_entries=45227),
             note="CCD molecule library, extracted to <cache>/mols"),

    # -- Protenix ---------------------------------------------------------------
    # v1 is upstream's own canonical v0.5.0 object. The `url` is what
    # `protenix/web_service/dependency_url.py` names at tag v0.5.0; the pxdesign bucket
    # serves the same bytes but sits in the same Volcengine region in Beijing, so "use
    # the other one" is not a fallback at all, and a customer install sat at 0 bytes
    # because of it. `moritztng/protenix-v0.5.0` on the hub is that file unmodified,
    # published with upstream's LICENSE and a card naming this URL, and is tried first.
    # The sha256 is of the finished file, measured on a fresh upstream download that
    # compared byte for byte with what the hub now serves, so a truncated download is a
    # named error rather than a crash inside torch.load three minutes later.
    Artifact("protenix-v1", ("protenix-v1",), "file", "Apache-2.0 (ByteDance)",
             repo=PROTENIX_V1_REPO, filename="model_v0.5.0.pt",
             url="https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.5.0.pt",
             sha256="9ea20b0aba42f2256711da1d0cd081510a4b291e64375bff6b70ced70b87a5f1",
             subdir="protenix", approx_bytes=1474265486,
             legacy_env=("PROTENIX_V1_CKPT",),
             note="ByteDance Protenix v0.5.0 base checkpoint"),
    Artifact("protenix-v2", ("protenix-v2",), "file", "Apache-2.0",
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

    # -- Nesso-1 (Recursion, Apache-2.0) ---------------------------------------
    # An hf-repo pair rather than two hf-file rows, the same shape as the two OpenDDE rows:
    # `tt-bio affinity` loads both files out of the hub cache
    # (tt_bio/nesso1.py::Nesso1.from_pretrained for the weights,
    # tt_bio/nesso1_input.py::find_ccd for the CCD), so an hf-file row would download
    # them to a flat
    # <cache>/ path nothing reads and the status table would report missing on a host that
    # has them. One snapshot of the pinned tag brings both, plus the 1.7 KB hparams.json
    # that sits beside the weights and needs no row of its own.
    #
    # ESM-2 650M deliberately gets no row: tt_bio/nesso1_input.py::run_esm reaches it through the
    # vendored `setup_esm_model`, i.e. through `transformers`, not through this fetch path.
    Artifact("nesso1", ("nesso1",), "hf-repo", "Apache-2.0 (Recursion)",
             repo=NESSO_REPO, revision=NESSO_REVISION,
             filename=f"{NESSO_REVISION}/model.safetensors", approx_bytes=165426752,
             note="affinity head; hparams.json sits beside it under the same revision tag"),
    Artifact("nesso1-ccd", ("nesso1",), "hf-repo", "Apache-2.0 (Recursion)",
             repo=NESSO_REPO, revision=NESSO_REVISION, filename="ccd.pkl",
             approx_bytes=412923533,
             note="CCD molecule dict, read by the host featurizer; $NESSO_CACHE also finds it"),

    # -- BoltzGen: six flat files under <cache>/boltzgen -----------------------
    Artifact("boltzgen-diverse", ("boltzgen",), "file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltzgen1_diverse.ckpt", subdir="boltzgen",
             approx_bytes=1930847192),
    Artifact("boltzgen-adherence", ("boltzgen",), "file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltzgen1_adherence.ckpt", subdir="boltzgen",
             approx_bytes=1930858014),
    Artifact("boltzgen-ifold", ("boltzgen",), "file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltzgen1_ifold.ckpt", subdir="boltzgen",
             approx_bytes=12582656),
    Artifact("boltzgen-folding", ("boltzgen",), "file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltz2_conf_final.ckpt", subdir="boltzgen",
             approx_bytes=2087255089),
    Artifact("boltzgen-affinity", ("boltzgen",), "file", "MIT",
             repo=BOLTZGEN_REPO, filename="boltz2_aff.ckpt", subdir="boltzgen",
             approx_bytes=2061914091),
    Artifact("boltzgen-mols", ("boltzgen",), "file", "MIT",
             repo=BOLTZGEN_REPO, filename="mols.zip", subdir="boltzgen",
             approx_bytes=391401102, note="read as a zip, not extracted"),

    # -- IPD direct downloads --------------------------------------------------
    Artifact("rf3", ("rf3",), "file", "see files.ipd.uw.edu (Institute for Protein Design)",
             url=f"{IPD_BASE}/rf3/rf3_foundry_01_24_latest_remapped.ckpt", subdir="rf3",
             approx_bytes=3038876446, legacy_env=("RF3_CKPT",)),
    Artifact("rfd3", ("rfd3",), "file", "see files.ipd.uw.edu (Institute for Protein Design)",
             url=f"{IPD_BASE}/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt", subdir="rfd3",
             approx_bytes=2690316669,
             derived=Derived("rfd3/weights", "rfd3", discard_archive=True, expect=(
                 "token_initializer.real_weights.pt", "token_initializer.real_weights.meta.json",
                 "diffusion_module.real_weights.pt", "diffusion_module.real_weights.meta.json")),
             note="only the extracted TokenInitializer/DiffusionModule weights are kept"),

    # -- PXDesign (ByteDance, Apache-2.0) --------------------------------------
    # Upstream URLs rather than a mirror of ours, the same way rf3/rfd3 point at IPD. The
    # origin serves slowly from outside CN; scripts/pxdesign_port/fetch_release_data.sh
    # exists for a parallel-range fetch when that matters.
    #
    # Only the generator is needed to run `tt-bio design --model pxdesign`. The Protenix
    # filter checkpoint and the CCD pair belong to the selection stages, which are not on
    # the CLI, so they are not listed as pxdesign artifacts and are not prefetched.
    Artifact("pxdesign", ("pxdesign",), "file", "Apache-2.0 (ByteDance)",
             url=f"{PXDESIGN_BASE}/release_model/pxdesign_v0.1.0.pt", subdir="pxdesign",
             approx_bytes=556554618, legacy_env=("PXDESIGN_CKPT",),
             note="PXDesign-d generator; the selection stages are not wired to the CLI"),

    # -- AlphaFold2 monomer pTM parameters (DeepMind, CC BY 4.0) ---------------
    # AF2-IG scores designs against these. Pinned to the 2022-12-06 release, VERIFIED rather
    # than assumed: the member's sha256 out of that tar is
    # 5e564f79af5bcd54ccef6e2a6bb0ff01015d01650ebc41d4575e35f0de9ecc84, which is byte for byte
    # what every committed AF2-IG tap and device floor was measured against. The file's own
    # mtime reads Jul 2021 and that is NOT the release -- it is the timestamp archived inside
    # the tar, preserved because these parameters were not rebuilt between the two releases.
    # Its own model key, not pxdesign's: the generator never loads these, only the AF2-IG
    # selection stage and the gate's two trunk legs do, so `tt-bio design --model pxdesign`
    # must not pull 4 GB on first use. `tt-bio weights --download af2ig` fetches it.
    Artifact("af2-params", ("af2ig",), "file", "CC BY 4.0 (DeepMind)",
             url="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
             subdir="af2", approx_bytes=4670017536, legacy_env=("AF2IG_PARAMS",),
             derived=Derived("af2/params", "af2-params", discard_archive=True,
                             expect=("params_model_1_ptm.npz",)),
             note="only params_model_1_ptm.npz is kept out of the 4 GB archive"),

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

def _bounded(fn, seconds: float, default):
    """Run ``fn()`` but stop waiting after ``seconds``.

    urllib applies its timeout per resolved address, and this bucket answers with nine
    A records, so a "15 second" HEAD against a host that drops packets takes 135 s of
    silence. The socket work is left to finish in a daemon thread; what matters is that
    nobody is still waiting on it."""
    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=seconds)
    except Exception:                                        # noqa: BLE001
        return default
    finally:
        pool.shutdown(wait=False)


def remote_size(url: str, timeout: float = 15.0, deadline: float = 30.0) -> int | None:
    """Content-Length for ``url``, or None if the server will not say in time.

    A byte-exact size from the source of record is the strongest completeness check
    there is, and unlike the archive-structure check it works on any file type. That
    matters for the MSA database tarballs, which are far too large to scan."""
    import urllib.request

    def ask():
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                n = resp.headers.get("Content-Length")
                return int(n) if n else None
        except Exception:                                    # noqa: BLE001
            return None
    return _bounded(ask, deadline, None)


def probe(url: str, *, timeout: float = 10.0, deadline: float = 20.0,
          sample_bytes: int = 1 << 20) -> dict:
    """Can this host be reached from here, and how fast? One ranged GET, never a file.

    The question that matters is not "does the name resolve" but "do bytes actually
    arrive", because the failure this exists for is a host that completes the TCP
    handshake and then sends nothing. So it asks for a megabyte and times it, and it
    gives up at ``deadline`` so a diagnosis is not itself something you wait on."""
    import urllib.error
    import urllib.request
    from urllib.parse import urlparse

    host = urlparse(url).netloc
    t0 = time.monotonic()

    def ask():
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes=0-{sample_bytes - 1}"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read(sample_bytes), ""
        except urllib.error.HTTPError as e:
            return e.code, b"", f"HTTP {e.code}"
        except Exception as e:                               # noqa: BLE001
            return None, b"", f"{type(e).__name__}: {e}"

    status, data, error = _bounded(ask, deadline,
                                   (None, b"", f"no answer within {deadline:.0f}s"))
    out = {"url": url, "host": host, "status": status, "bytes": len(data),
           "ok": bool(data), "error": error,
           "seconds": round(time.monotonic() - t0, 2), "mb_per_s": 0.0}
    if out["bytes"] and out["seconds"]:
        out["mb_per_s"] = round(out["bytes"] / out["seconds"] / _MB, 2)
    if not out["ok"] and not out["error"]:
        out["error"] = "connected but sent no data"
    return out


def sha256_of(path: Path) -> str:
    """sha256 of a file, read in 4 MiB blocks (1.4 GiB in ~5 s)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Downloading: bounded, mirror-first, and loud about what it tried
# ---------------------------------------------------------------------------

# A transfer that has not moved a byte for this long is dead, whatever the tool
# believes. Every download tool has its own idea of a timeout and some of them
# effectively have none, so the bound that actually holds is ours: watch the bytes
# landing on disk and cut the tool off when they stop. Without it, a host that
# accepts the connection and then sends nothing parks the caller forever. That is
# exactly what a customer's portal did -- aria2c printing "0B/0B" against a
# China-hosted bucket her network could not pull from, with aria2c's max-tries set
# to zero, which is its word for "retry forever", inside a subprocess.run() with no
# timeout.
STALL_SECONDS = 120
# A source that has not produced a single byte in this long is not going to. Kept well
# under STALL_SECONDS because it is a different question: "this transfer has stopped"
# deserves patience, "this host has never sent us anything" does not, and the whole
# point is that she finds out in minutes instead of never.
NO_START_SECONDS = 60
CONNECT_TIMEOUT = 20

_progress_hook = None


@contextmanager
def progress_to(fn):
    """Report download progress to ``fn(name, done_bytes, total_bytes)`` in here.

    A scoped module hook rather than an argument on every fetch: the portal's device
    worker reaches a download four frames below ``_ensure_local_artifacts``, and
    threading a callback through each of them to serve one caller is worse than one
    hook that is set and unset around the call."""
    global _progress_hook
    previous = _progress_hook
    _progress_hook = fn
    try:
        yield
    finally:
        _progress_hook = previous


def _report(name: str, done: int, total: int | None) -> None:
    if _progress_hook is None:
        return
    # The watcher polls the staging file, but the person reading the UI is waiting for
    # the checkpoint, so it is named ".model_v0.5.0.pt.part" here and model_v0.5.0.pt
    # everywhere they can see.
    if name.startswith(".") and name.endswith(".part"):
        name = name[1:-len(".part")]
    try:
        _progress_hook(name, done, total)
    except Exception:                                        # noqa: BLE001
        pass          # a UI that cannot be told is never a reason to fail a download


class DownloadFailed(RuntimeError):
    """Every source for one file was tried and none delivered it.

    Carries the attempt table, so the caller can say which host was tried, with what,
    how far it got and what to do next. "Please retry" is what made the last one a
    round of email."""

    def __init__(self, name: str, attempts: list[tuple[str, str, str, int]]):
        self.name = name
        self.attempts = attempts
        lines = [f"could not download {name}. Tried, in order:"]
        for url, tool, reason, got in attempts:
            lines.append(f"    {tool:<7} {source_url(url)}")
            lines.append(f"            -> {reason}"
                         f"{'' if got is None else f' after {got} byte(s)'}")
        super().__init__("\n".join(lines))

    @property
    def bytes_received(self) -> int:
        return max((got or 0) for _u, _t, _r, got in self.attempts) if self.attempts else 0


class WeightsUnavailable(RuntimeError):
    """One model's weights could not be obtained, said so that the next step is obvious.

    The person reading this is on a machine we cannot log in to, so the message has to
    carry everything the next action needs: which hosts were tried, whether any byte
    arrived (a network problem, not a corrupt file), where to put the file by hand, how
    big it is, its sha256, and the env var that points at a copy elsewhere."""

    def __init__(self, art: Artifact, failure: DownloadFailed, root=None):
        self.artifact = art
        self.failure = failure
        dest = art.dest(root)
        got = failure.bytes_received
        nothing = got == 0
        lines = [
            f"{art.key} weights are missing and no source would serve them.",
            "",
            str(failure),
            "",
            ("Not one byte arrived, so this is the network on this machine, not a "
             "corrupt file." if nothing else
             f"{got} byte(s) arrived and then the transfer died."),
            "",
            "What to do:",
            f"  1. tt-bio preflight {art.models[0]}",
            "     says which of those hosts this machine can actually reach, and how fast.",
            "  2. If one is reachable, run it again: the download resumes where it stopped.",
            "  3. If none is reachable from here, fetch it on a machine that can and copy it:",
        ]
        for src in art.sources:
            lines.append(f"       {source_url(src)}")
        lines += [
            f"       {art.approx_bytes / _GB:.2f} GiB"
            + (f", sha256 {art.sha256}" if art.sha256 else ""),
            f"     put it at {dest}",
            f"     or point ${art.env} at wherever you put it.",
        ]
        super().__init__("\n".join(lines))


def _tool_commands(url: str, dest: Path) -> list[tuple[str, list[str]]]:
    """The download commands to try for one URL, best first, each one bounded.

    Bounded is the whole point. These ran with aria2c's max-tries set to zero, its
    word for "forever", and with no connect or read timeout on curl or wget. The stall
    watchdog below is the backstop that catches whatever they still ignore; these
    flags keep a dead connection from spending the whole budget before it."""
    tools: list[tuple[str, list[str]]] = []
    if shutil.which("aria2c"):
        tools.append(("aria2c", [
            "aria2c", "--max-connection-per-server=8", "--split=8",
            "--continue=true", "--auto-file-renaming=false",
            # The watchdog below reads the size of the staging file, so that size has
            # to mean "bytes received". Preallocation sizes the file to the full 1.4 GB
            # up front, which makes "not one byte has arrived" indistinguishable from
            # "it is all here" and costs the 60 s no-start rule its only signal.
            "--file-allocation=none",
            f"--connect-timeout={CONNECT_TIMEOUT}", "--timeout=60",
            "--max-tries=3", "--retry-wait=5", "--lowest-speed-limit=1K",
            "--summary-interval=15", "-o", dest.name, "-d", str(dest.parent), url]))
    if shutil.which("curl"):
        # -f, so an HTML 404 body from a mistyped mirror is a failure rather than a
        # 500-byte "checkpoint" that dies later inside torch.load.
        tools.append(("curl", [
            "curl", "-fL", "--retry", "3", "--retry-delay", "5",
            "--connect-timeout", str(CONNECT_TIMEOUT),
            "--speed-limit", "1024", "--speed-time", "60",
            "-C", "-", "--progress-bar", "-o", str(dest), url]))
    if shutil.which("wget"):
        tools.append(("wget", [
            "wget", "-c", "--tries=3", "--waitretry=5",
            f"--connect-timeout={CONNECT_TIMEOUT}", "--read-timeout=60",
            "-O", str(dest), url]))
    return tools


def _run_tool(name: str, cmd: list[str], watch: Path, *, total: int | None,
              quiet: bool) -> tuple[bool, str, int, bool]:
    """Run one download command, killing it when the bytes stop arriving.

    Returns (succeeded, reason, bytes on disk, source looks dead). The watchdog polls
    the file rather than parsing the tool's progress output, so it works the same for
    aria2c, curl, wget and anything added later. "Source looks dead" means not one
    byte ever arrived, which is the caller's cue to stop trying this URL at all rather
    than repeat the same silence with the next tool."""
    import signal
    import subprocess

    def size() -> int:
        try:
            return watch.stat().st_size
        except OSError:
            return 0

    start = last = size()
    moved_at = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL if quiet else None,
                            stderr=subprocess.STDOUT if quiet else None,
                            start_new_session=True)
    while True:
        try:
            rc = proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            rc = None
        now = size()
        if now != last:
            last, moved_at = now, time.monotonic()
            _report(watch.name, now, total)
        if rc is not None:
            return rc == 0, ("finished" if rc == 0 else f"exit code {rc}"), now, False
        silent = now == start == 0
        if time.monotonic() - moved_at > (NO_START_SECONDS if silent else STALL_SECONDS):
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except OSError:
                    break
                try:
                    proc.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if silent:
                return False, f"nothing arrived in {NO_START_SECONDS}s", 0, True
            return False, f"no data for {STALL_SECONDS}s", now, False


def _hf_download_to(repo: str, filename: str, dest: Path, *, quiet: bool,
                    log: list) -> tuple[bool, bool]:
    """Fetch one file out of a Hugging Face repo into ``dest``. Never a final path.

    The hub client does its own staging and its own retries, and every request it makes
    carries a 10 s read timeout (``huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT``),
    so a host that goes quiet mid-transfer raises here instead of parking us: this
    source needs no stall watchdog of its own, only the same "one attempt, then the
    next source" rule as the tools. It downloads into a scratch dir beside ``dest``
    rather than the hub cache, so a flat row keeps one copy on disk rather than two."""
    import tempfile

    _echo(f"  Downloading {dest.name} from {repo} ...", quiet)
    _report(dest.name, 0, None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sweep_stale_staging(dest.parent)
    staging = Path(tempfile.mkdtemp(dir=str(dest.parent), prefix=".dl-"))
    try:
        from huggingface_hub import hf_hub_download
        tmp = Path(hf_hub_download(repo_id=repo, filename=filename,
                                   local_dir=str(staging), force_download=True))
        os.replace(tmp, dest)
        log.append((f"{_HF_SCHEME}{repo}/{filename}", "hf", "finished",
                    dest.stat().st_size))
        return True, False
    except Exception as e:                                   # noqa: BLE001
        log.append((f"{_HF_SCHEME}{repo}/{filename}", "hf", f"{type(e).__name__}: {e}", 0))
        _echo(f"    hf: {type(e).__name__}: {e}", quiet)
        return False, False
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _download_to(source: str, dest: Path, *, quiet: bool = False,
                 attempts: list | None = None) -> tuple[bool, bool]:
    """Fetch one source into ``dest`` with tool fallback. Never called with a final path.

    Returns (a tool reported success, this source is worth another try). Every attempt
    is appended to ``attempts`` so a failure can say exactly what it tried."""
    import urllib.request

    log = attempts if attempts is not None else []
    if (hf := split_hf(source)) is not None:
        return _hf_download_to(*hf, dest, quiet=quiet, log=log)
    url = source
    total = remote_size(url, deadline=25.0)
    _echo(f"  Downloading {dest.name} from {url.split('/')[2]} ...", quiet)
    _report(dest.name, dest.stat().st_size if dest.exists() else 0, total)
    tools = _tool_commands(url, dest)
    if not tools:
        _echo("    (no aria2c/curl/wget, using Python urllib, may be slow)", quiet)
        try:
            urllib.request.urlretrieve(url, dest)
            return True, True
        except Exception as e:                               # noqa: BLE001
            log.append((url, "urllib", f"{type(e).__name__}: {e}", 0))
            return False, False
    for name, cmd in tools:
        good, reason, got, dead = _run_tool(name, cmd, dest, total=total, quiet=quiet)
        log.append((url, name, reason, got))
        if good:
            return True, True
        _echo(f"    {name}: {reason}", quiet)
        if dead:
            # Silence is a property of the host, not of the tool. Handing the same
            # dead URL to curl and then wget only spends the user's afternoon.
            return False, False
    return False, True


def fetch_file(sources: tuple[str, ...], dest: Path, *, sha256: str | None = None,
               force: bool = False, quiet: bool = False,
               check_archive: bool = True) -> Path:
    """Download the first of ``sources`` that delivers to ``dest``, atomically.

    One path for every flat artifact, whatever transport it comes over: an
    ``hf://repo/file`` source goes through the hub client, anything else through
    aria2c/curl/wget. Sources are tried in order, so an origin this machine cannot
    reach is a delay rather than a dead end.

    Staging is a stable ``.<name>.part`` next to the destination so an interrupted
    multi-GB download resumes instead of restarting, while the destination itself only
    ever holds a verified file. ``check_archive=False`` skips the structural check for a
    file too large to scan (the MSA databases), leaving the byte count to carry it.

    Raises ``DownloadFailed``, which names every source and tool that was tried."""
    dest = Path(dest)
    sources = tuple(s for s in sources if s)
    if not sources:
        raise ValueError(f"no source for {dest.name}")

    def cached_ok(path: Path) -> bool:
        """Is what is already on disk usable? No network: this runs before every
        fold, and it used to cost a HEAD request to the origin each time."""
        try:
            if path.stat().st_size == 0:
                return False
        except OSError:
            return False
        if check_archive:
            return artifact_intact(path, dest.name)
        expect = next((remote_size(s) for s in sources if not split_hf(s)), None)
        return expect is None or path.stat().st_size == expect

    def fresh_ok(path: Path, source: str) -> tuple[bool, str]:
        """Is what just arrived the file we asked for, and if not, in which way?

        The strongest check we have, paid once per download rather than once per fold.
        Each answer is a different failure and gets its own words, because this string
        is what the person on the other machine reads."""
        size = path.stat().st_size if path.exists() else 0
        if size == 0:
            return False, "nothing arrived"
        if check_archive and not artifact_intact(path, dest.name):
            return False, "truncated or not a readable archive"
        if sha256:
            got = sha256_of(path)
            if got != sha256:
                return False, f"sha256 {got[:16]}… != expected {sha256[:16]}…"
            return True, ""
        # No recorded hash: fall back to the server's own byte count. The hub client
        # already checks the blob against the repo's etag, so there is nothing left to
        # ask it.
        expect = None if split_hf(source) else remote_size(source)
        if expect is not None and size != expect:
            return False, f"{size} bytes, server says {expect}"
        return True, ""

    if not force and dest.exists():
        if cached_ok(dest):
            return dest
        _echo(f"Cached {dest.name} is incomplete/corrupt, re-downloading", quiet)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f".{dest.name}.part")

    attempts: list[tuple[str, str, str, int]] = []
    for source in sources:
        # Twice per source: a resume that lands on a mid-file corruption can never be
        # fixed by resuming, so the second try starts from scratch.
        for attempt in (1, 2):
            got_it, worth_retrying = _download_to(source, part, quiet=quiet,
                                                  attempts=attempts)
            if got_it:
                good, why = fresh_ok(part, source)
                if good:
                    os.replace(part, dest)
                    return dest
                attempts.append((source, "verify", why, part.stat().st_size
                                 if part.exists() else 0))
                _echo(f"    verify: {why}", quiet)
            part.unlink(missing_ok=True)
            if not worth_retrying:
                break
            if attempt == 1:
                _echo(f"    retrying {dest.name} from scratch", quiet)
    raise DownloadFailed(dest.name, attempts)


def fetch_hf_repo(repo_id: str, *, filename: str | None = None, revision: str | None = None,
                  force: bool = False, quiet: bool = False) -> Path:
    """Snapshot a whole HF repo into the hub cache and return the snapshot dir.

    ``revision`` pins a tag or commit. It is not cosmetic for a repo whose default
    branch is not the released tree: ``recursionpharma/nesso`` keeps ``main`` and
    ``v1.0.0`` on different commits, and ``main`` holds only a ``config.json``.

    The hub cache is already written blob-at-a-time through ``.incomplete`` staging, so
    the destination cannot hold a partial blob. What it does not do is notice a blob
    that went bad afterwards (a half-finished copy of the cache between hosts, a full
    disk during the final link), so when ``filename`` names the weight file we verify
    it and re-snapshot with ``force_download`` if it fails."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(repo_id, revision=revision, force_download=force))
    if filename:
        target = snap / filename
        if not target.exists() or not artifact_intact(target):
            _echo(f"Cached {repo_id}:{filename} is incomplete/corrupt, re-downloading", quiet)
            snap = Path(snapshot_download(repo_id, revision=revision, force_download=True))
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
    elif producer == "af2-params":
        # A member-filtered extraction, not the stock `tar` producer: the AlphaFold archive is
        # 4 GB of five model variants and AF2-IG reads exactly one of them, so extracting the
        # lot would leave 4 GB of parameters nothing loads.
        import tarfile
        _echo(f"Extracting params_model_1_ptm.npz from {archive.name}", quiet)
        want = "params_model_1_ptm.npz"
        staging.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            member = next((m for m in tar if Path(m.name).name == want), None)
            if member is None:
                raise ValueError(f"{archive.name} does not contain {want}")
            src = tar.extractfile(member)
            if src is None:
                raise ValueError(f"{want} in {archive.name} is not a regular file")
            with (staging / want).open("wb") as fh:
                shutil.copyfileobj(src, fh)
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
            hit = try_to_load_from_cache(art.repo, art.filename, revision=art.revision)
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
        snap = fetch_hf_repo(art.repo, filename=art.filename, revision=art.revision,
                             force=force, quiet=quiet)
        return snap / art.filename if art.filename else snap

    if art.source != "file":
        raise ValueError(f"unknown source {art.source!r} for {key}")

    path = art.dest(root)
    if art.derived:
        # The derived output is what the model reads, so a complete one means there is
        # nothing left to fetch. For RFD3 and the AF2 parameters the archive is deleted
        # after extraction and its absence is normal; for the CCD library it is simply
        # 1.8 GB nobody reads again. Skipping it here is also what makes `--download`
        # agree with the table, which already calls such a row `present`.
        out = cache_root(root) / art.derived.subdir
        if not force and (_marker(out).exists() and out.is_dir() or _derived_ok(out, art.derived)):
            return ensure_derived(path, art.derived, root=root, quiet=quiet)
        _echo(f"Downloading {art.key} "
              f"(~{art.approx_bytes / _GB:.1f} GiB, {source_host(art.sources[0])})", quiet)
    try:
        path = fetch_file(art.sources, path, sha256=art.sha256, force=force, quiet=quiet)
    except DownloadFailed as e:
        raise WeightsUnavailable(art, e, root) from None

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

def model_ready(model: str, root: str | Path | None = None) -> tuple[bool, list[Status]]:
    """Is every artifact this model loads present and intact, and what is not?

    One predicate, so `tt-bio preflight`, the portal's submit check and diagnose.sh
    cannot disagree about whether a model can run on this host."""
    stats = [status(a.key, root) for a in artifacts_for(model)]
    return all(s.state == "present" for s in stats), stats


def models_known() -> tuple[str, ...]:
    """Every model name the registry knows, sorted."""
    return tuple(sorted(MODEL_ARTIFACTS))


def artifact_paths(key: str, root: str | Path | None = None) -> list[Path]:
    """Everything on disk that belongs to one row: the archive, its derived output and
    the completion marker. Hub-cache rows return their snapshot dir, which is shared
    with the hub's own blob store, so pruning those goes through ``delete_revisions``
    rather than a plain unlink."""
    art = ARTIFACTS[key]
    out: list[Path] = []
    if art.source in ("file", "manual"):
        out.append(art.dest(root))
    if art.derived:
        d = art.derived_dest(root)
        out += [d, _marker(d)]
    return [p for p in out if p.exists()]


def cached_path(key: str, root: str | Path | None = None) -> Path | None:
    """Where this artifact already IS on disk, or None. Never downloads.

    ``artifact_paths`` answers the disk-audit question and returns [] for an ``hf-repo`` row,
    whose file lives in the hub's snapshot tree rather than the flat cache. That is correct for
    an audit and wrong for "do I have this?", and a test that reached for ``Artifact.dest()``
    instead -- documented as meaningless for hf-repo -- silently SKIPped every hf-repo model it
    was supposed to check. A guard that does not run looks exactly like a guard that passed, so
    the predicate belongs here once rather than in each caller.
    """
    art = ARTIFACTS[key]
    if (p := _override(art)) is not None:
        return p if p.exists() else None
    if art.derived:
        d = art.derived_dest(root)
        return d if d and d.exists() else None
    if art.source == "hf-repo":
        if not art.filename:
            return None
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return None
        configure_hf_cache()
        hit = try_to_load_from_cache(art.repo, art.filename)
        return Path(hit) if isinstance(hit, str) else None
    p = art.dest(root)
    return p if p.exists() else None


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
