"""AbAg-XM Tier-A generation: fold every 2026ARK-AB target (164) across the three
generators (protenix-v2, opendde-abag, boltz2) at 50 diffusion samples with
--write_pae, fully offline against the local ColabFold DB. One JSON line per
(target, model) is appended to progress.jsonl so partial progress survives a
restart; a (target, model) already recorded status=ok is skipped.

Fan across cards by passing a target subset per invocation (one card per
invocation, one device context per process):

    TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_HOLDER=worker:abag-xm-crossmodel-ranking-dataset-p3 \
    PYTHONPATH=$PWD python3 scripts/abag_xm_generate.py --targets 9w14,9j4c --device <card>

Per-sample DockQ against the ARK interface is NOT computed here -- it is a CPU label
(Phase 4), not device work; this phase only produces the 50 CIFs + per-sample PAE
artifacts that Phase 4 scores. The failed-results cross-check (results.json status
must be "ok" AND all_runs/n_cifs must equal n_samples) is enforced so a job whose
results.json says failed is never recorded ok (opendde-abag-paired-msa-offline-gap).
"""
import argparse, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
# colabfold_search (invoked by tt_bio for uncached MSAs) needs localcolabfold's
# own mmseqs (v18, supports --prefilter-mode); the apt mmseqs (v13) exits 1.
_lc_bin = str(Path.home() / "localcolabfold" / ".pixi" / "envs" / "default" / "bin")
if Path(_lc_bin).exists():
    os.environ["PATH"] = _lc_bin + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parent.parent
# Persistent (never /tmp -- qb1 clears it, losing hours of fold work). All tiers/hosts
# share one MSA cache (keyed by sequence hash) and one offline DB.
OUT_BASE = Path.home() / "abag_xm" / "tier_a"
MSA_DIR = Path.home() / "abag_xm" / "msa_cache"
MSA_DB_PATH = Path.home() / ".boltz" / "msa_db"
YAML_DIR = ROOT / "examples" / "abag_xm"
# Ground-truth reference structures. 143.8 MiB of append-only mmCIF has no business in git
# (they ship as a Release asset), so prefer the host data directory and fall back to whatever
# the checkout happens to carry.
GT_HOST = Path.home() / "abag_xm" / "ground_truth"
GT = GT_HOST if GT_HOST.is_dir() else ROOT / "examples" / "ground_truth_structures"
PROGRESS = OUT_BASE / "progress.jsonl"
# Local mirror of the OTHER host's progress file, refreshed by abag_xm_peer_mirror.sh. Scheduling
# input only -- never a record, never evidence. See peer_done_pairs().
PEER_PROGRESS = OUT_BASE / "peer_progress.jsonl"
# Per-fold provenance, written beside the output BEFORE the fold runs. The orphan reconciler
# reads this rather than inferring a dead fold's engine tree -- see write_fold_provenance().
FOLD_PROVENANCE = ".fold_provenance.json"
MANIFEST = ROOT / "docs" / "implementation-parity-data" / "abag-xm-targets.parquet"

MODELS = ["protenix-v2", "opendde-abag", "boltz2"]
RESULT_PREFIX = {"protenix-v2": "protenix", "opendde-abag": "opendde",
                 "opendde": "opendde", "boltz2": "boltz2"}
# Tier-A config: 50 samples, seed grid. boltz2 uses mps=5 (the fixed, memory-bounded
# chunking); protenix/opendde use their own (correct) sampler at the same mps=5.
N_SAMPLES = 50
SEED = 42
# The parked pre-p4 slab measured the worst case directly: 9j4c (1095 tokens, the largest target)
# took 3238 s = 54 min for 50 samples on the sequential path. Against the old 3600 s that is a 10%
# margin, and 9j4c / 9i3p / 9ivj each appear BOTH ok and timed_out in that slab -- i.e. they sat on
# the timeout boundary and flipped with host contention. That, not "slow targets", is why the three
# largest protenix targets were recorded timed_out. s/token is stable at 2.7-3.4 across the top end,
# so the ceiling is predictable rather than pathological. 7200 s is ~2.2x the observed worst case and
# matches what abag_xm_resume_opendde.sh already uses.
FOLD_TIMEOUT_S = 7200
# A card whose firmware will not initialise fails every fold at device-open, long before any
# model work runs. 25 s was the measured cost of such a fold; the fastest real fold is ~300 s,
# so 120 s separates the two by a wide margin and cannot misfire on a genuinely quick fold.
DEAD_CARD_MAX_S = 120
# One fast failure is a transient (a sibling briefly holding the card); three in a row is a chip.
DEAD_CARD_STREAK = 3
# Per-fold timeout scaled to the target instead of a flat cap. Measured ceilings in
# s/residue on qb1 (the slower host): the MAX observed across completed folds, not the
# median -- protenix 2.5-4.0, opendde 2.8-4.0, boltz2 0.76-0.88. A flat 7200 s is ~20x a
# boltz2 fold's real cost, which is why a stalled fold burns two card-hours before anything
# notices; three have (22ps protenix, 9i5n boltz2, 9iar boltz2 -- one spinning dispatch
# thread at 100% CPU, no output written, deaf to SIGINT and SIGTERM). Scaling catches the
# same stall in minutes. The result is clamped to FOLD_TIMEOUT_S so it can only ever be
# TIGHTER than today, never looser, and the floor keeps small targets generous.
RATE_CEILING_S_PER_RES = {"protenix-v2": 4.0, "opendde-abag": 4.0, "boltz2": 0.9}
# boltz2 only: when max_parallel_samples does not divide diffusion_samples the chunk split is
# ragged, which issues ~1.7x the per-step dispatch and measured 2.23 s/res against 0.88
# uniform. Using the uniform ceiling there leaves 1.4x headroom -- tight enough to kill a
# healthy fold -- and qb2 is still running that configuration.
BOLTZ2_RAGGED_CEILING_S_PER_RES = 2.6
TIMEOUT_MARGIN = 3.5           # >=3.4x headroom over the slowest fold ever observed at size
FOLD_TIMEOUT_FLOOR_S = 900
# max_parallel_samples. BACK TO 5, the only value validated for boltz2.
# 3 was set while chasing the protenix OOM, on a hypothesis that was then retracted: the
# protenix batched path is entered on n_sample > 1 regardless of mps, so mps never affected that
# OOM, and with supports_multiplicity=False protenix/opendde ignore mps entirely. So this knob now
# only reaches boltz2 -- the one generator that genuinely chunks on it (boltz2.py:4074-4147).
# At mps=3 boltz2 dies in ~68 s with
#   TT_FATAL binary_ng_device_operation.cpp:346: a_dim == b_dim || a_dim == 1 || b_dim == 1
#   Broadcasting rule violation
# while the parked slab has 65 boltz2 ok records at mps=5 (its 130 failures were the
# interpreter bug, not mps). The mechanism at 3 is NOT established; 5 is simply the validated
# value and the campaign has no reason to deviate from it.
MPS = 5
# Root-caused and fixed in 151400b8 (the device conditioning cache kept the first chunk's
# sample batch, so a short final chunk mismatched it) -- but that fix is NOT yet verified on
# a card, so MPS stays at the validated value and the guard below keeps any fold generated
# under a different one out of the slab.
MPS_SENSITIVE_MODELS = {"boltz2"}  # the others ignore mps (supports_multiplicity=False)
CONCURRENT_FOLDS = 4  # one harness per card on a 4-card QuietBox; sets the per-fold CPU share
# How long a fold waits for its own card before giving up (tt_bio's TT_BIO_LEASE_TIMEOUT,
# default 120 s). 900 s is about the longest fold this campaign runs, so a card taken
# legitimately by a sibling is waited out rather than turned into a failed record.
LEASE_TIMEOUT_S = 900

# D12 (per-model config fairness contract) — resolved-config fields recorded in
# every progress.jsonl line so Phase 4 can assert constancy within a generator
# before any table is produced. paired_msa=False for all three: no generator
# receives a species-paired MSA in this campaign (offline pairing is not available
# without a `:`-joined complex query; see opendde-abag-paired-msa-offline-gap).
_HOST = socket.gethostname()


def _head_commit():
    """The worktree's HEAD right now, suffixed ``-dirty`` if it has uncommitted changes.

    Read per fold, not once at import: each fold is a fresh interpreter that loads
    whatever is in the worktree when it launches, so a value captured at startup
    misdescribes every fold started after a commit -- and commits do land mid-campaign.
    """
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, timeout=30).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True, timeout=30).strip()
        return f"{head}-dirty" if dirty else head
    except Exception:
        return "unknown"


def _head_tree():
    """Hash of the ``tt_bio/`` subtree at HEAD -- the only thing that can move a fold number.

    ``tt_bio_commit`` counts commits to anything in the repo, including this harness and the
    label scripts, neither of which the folding interpreter loads. Over the pre-p6 slab that
    made 34 commit strings look like 34 procedures when they were 5 engine trees differing
    only in device-open and CLI code. Record the subtree and engine constancy becomes a
    one-line assertion instead of archaeology.

    Returns None when it cannot be stated (dirty worktree, detached rev-parse failure). A
    None here means the fold must not run -- see ``provenance_or_die``.
    """
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "tt_bio"],
            cwd=ROOT, text=True, timeout=30).strip()
        if dirty:
            return None
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD:tt_bio"], cwd=ROOT, text=True, timeout=30).strip()
    except Exception:
        return None


def _msa_sha(target):
    """sha256 over the cached MSA bytes this target's fold will actually consume.

    The engine picks the MSA for a chain by ``sha256(seq)[:16] + ".a3m"`` in ``--msa_dir``, and
    falls through to building one with whichever ``mmseqs`` it can find when the file is absent.
    That fall-through is the one path in the harness that can change a fold's *input*, so
    argue about it with bytes: hash the a3m files in cache order and record the digest. Two
    folds with the same engine tree and the same digest consumed the same input, full stop.

    Returns (digest, n_chains_missing). A missing chain is not fatal here -- it is the MSA
    cache miss that produced six ``incomplete`` records, and it is recorded so a resume can
    see it rather than rediscovering it from a traceback.
    """
    import hashlib
    try:
        import yaml as _yaml
        doc = _yaml.safe_load((YAML_DIR / f"{target}.yaml").read_text()) or {}
    except Exception:
        return None, -1
    h, missing = hashlib.sha256(), 0
    for s in doc.get("sequences", []) or []:
        p = s.get("protein", {}) if isinstance(s, dict) else {}
        seq = p.get("sequence") or ""
        if not seq:
            continue
        f = MSA_DIR / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m"
        if f.exists():
            h.update(f.read_bytes())
        else:
            missing += 1
            h.update(b"\0MISSING\0")
    return h.hexdigest()[:16], missing


def provenance_or_die():
    """Abort before folding anything if this slice's numbers could not be defended later.

    Fold-then-record-``unknown`` is worse than not folding: the record is indistinguishable
    from a good one in every downstream table, and 19 folds in the pre-p6 slab reached exactly
    that state (7 ``-dirty``, 12 ``None``) where the release preflight blocks on them and the
    resume filter skipped them forever. Fail loudly at startup instead -- a dirty ``tt_bio/``
    is a one-command fix and the operator is right there.
    """
    tree = _head_tree()
    if tree:
        return tree
    raise SystemExit(
        "abag_xm_generate: refusing to fold -- the tt_bio/ engine tree cannot be stated.\n"
        f"  worktree: {ROOT}\n"
        "  Either tt_bio/ has uncommitted changes (commit or stash them) or `git rev-parse\n"
        "  HEAD:tt_bio` failed. Every record this slice wrote would carry unstateable\n"
        "  provenance, which the release preflight rejects and the resume filter cannot skip.")


# --- infrastructure failures are not fold results ---------------------------------
# A fold that never reached the model because the card was unavailable is a scheduling
# event, not a measurement. Writing it as `fold_failed` is what made 37 records look like a
# model regression: two distinct mechanisms, both ending at device-open.
#
#   A1  another process holds the card. tt_bio's DeviceLease polls flock for
#       TT_BIO_LEASE_TIMEOUT (120 s by default) and then raises DeviceInUseError, so the
#       fold dies at a near-constant 133 s = 120 + startup. The fleet dispatcher samples a
#       multi-card campaign's brief inter-fold gap as a free card and takes it.
#   A2  the chip will not initialise. No lease wait at all -- device-open fails outright in
#       ~23 s. One wedged card silently eats its whole slice.
#
# Both are retryable, and retrying is the only way the campaign reaches zero failures: the
# card comes back (A1 after the thief finishes, A2 after a reset), and the same fold then
# produces a real number. Only an exhausted retry budget is a record.
INFRA_SIGNATURES = (
    "DeviceInUseError",
    "another process already holding the card",
    "all local workers exited before the run finished",
    "Failed to open device",
    "Device is in use",
)
INFRA_RETRIES = 4           # 4 retries x 300 s backoff covers a ~20 min steal
INFRA_BACKOFF_S = 300
DEAD_CARD_RESETS = 2        # per slice; beyond this the chip needs a human, not another reset


def _is_infra_failure(rec):
    """True iff this record is a card-availability event rather than a fold result."""
    if rec.get("status") not in ("fold_failed", "killed"):
        return False
    err = rec.get("stderr") or ""
    return any(s in err for s in INFRA_SIGNATURES)


def _reset_card(device):
    """tt-smi -r on ONE card. Scoped deliberately: a QuietBox is shared with the fleet and
    resetting the whole board would hard-reset a sibling worker's in-flight card."""
    tt_smi = Path.home() / ".tenstorrent-venv" / "bin" / "tt-smi"
    if not tt_smi.exists():
        tt_smi = Path.home() / ".local" / "bin" / "tt-smi"
    if not tt_smi.exists():
        print(f"[reset] no tt-smi found -- card {device} needs a manual reset", flush=True)
        return False
    print(f"[reset] tt-smi -r {device}", flush=True)
    try:
        r = subprocess.run([str(tt_smi), "-r", str(device)], capture_output=True,
                           text=True, timeout=180)
        ok = r.returncode == 0
        print(f"[reset] card {device} rc={r.returncode}", flush=True)
        time.sleep(10)
        return ok
    except Exception as e:
        print(f"[reset] card {device} failed: {e}", flush=True)
        return False


# --- fold interpreter -------------------------------------------------------------
# NEVER use bare sys.executable here. This script's own usage line launches it as
# `PYTHONPATH=$PWD python3 scripts/abag_xm_generate.py`, i.e. SYSTEM python, which cannot
# import tt-bio's deps -- so every fold died instantly with
# `ModuleNotFoundError: No module named 'click'` while still being recorded as a
# progress.jsonl entry. That is how 270 of 438 Tier-A records became fold_failed while the
# campaign looked like it was progressing (audit 2026-07-27). The shell wrappers
# (abag_xm_tiera_launch.sh) hardcodes the venv python, which is why the folds launched
# through them are the ones that worked.
# Resolve a python that can actually import the deps, and VALIDATE ONCE up front so a bad
# interpreter is a loud startup abort instead of N silent per-fold failures.
def _resolve_fold_python():
    """Pick an interpreter that can actually RUN a fold, preferring the tt-bio venvs.

    Validation is `-m tt_bio.main --help`, not `import click, tt_bio`: the import check is too
    weak -- on some hosts the SYSTEM python can import both yet still lack torch/ttnn, so it
    would be selected and then fail at fold time, which is the exact silent-failure class this
    guard exists to prevent. The venvs are tried BEFORE sys.executable for the same reason.
    """
    cands = [str(ROOT / "env" / "bin" / "python3"),
             str(Path.home() / "tt-bio" / "env" / "bin" / "python3"),
             str(Path.home() / "tt-bio-dev" / "env" / "bin" / "python3"),
             sys.executable]
    seen, tried = set(), []
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
        if not Path(c).exists():
            tried.append(f"{c} -> absent")
            continue
        r = subprocess.run([c, "-m", "tt_bio.main", "--help"], capture_output=True, text=True,
                           cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
                           timeout=300)
        last = (r.stderr or "").strip().splitlines()
        tried.append(f"{c} -> rc={r.returncode}" + (f" ({last[-1][:110]})" if r.returncode and last else ""))
        if r.returncode == 0:
            return c
    raise SystemExit("abag_xm_generate: no interpreter can run `-m tt_bio.main --help`. Tried:\n  "
                     + "\n  ".join(tried))


FOLD_PY = _resolve_fold_python()

def all_targets():
    """164 target ids from the manifest, in manifest order (stable across relaunches)."""
    try:
        import pandas as pd
        df = pd.read_parquet(MANIFEST)
        return df["pdb_id"].tolist()
    except Exception:
        # fall back to the on-disk fold YAMLs if pandas/manifest unavailable
        return sorted(p.stem for p in YAML_DIR.glob("*.yaml"))


def _is_done_record(r):
    """Is this `ok` record an accepted fold? Must match `abag_xm_acceptance.py` exactly.

    The resume filter and the release gate had drifted apart, so a pair could be skipped as done
    and rejected as unpublishable at the same time -- which is how 21 pairs deadlocked. Keep the
    two definitions in step; the tests pin every way they can diverge.
    """
    if r.get("status") != "ok":
        return False
    if r["model"] in MPS_SENSITIVE_MODELS and r.get("mps") != MPS:
        # Generated under a different sampling configuration -- a different procedure, so not
        # done. The resume pass regenerates it with --override.
        return False
    c, t = r.get("tt_bio_commit"), r.get("tt_bio_tree")
    if not t and (not c or c.endswith("-dirty")):
        # Provenance cannot be stated, so the fold cannot go in a published slab: the release
        # preflight blocks on exactly these records. Treating them as done is a deadlock -- the
        # resume skips them forever while the gate waits for them.
        return False
    # The artifact counts too, for the same reason: a record claiming ok with n_samples+1 PAEs
    # was not written by this harness (its glob excludes the aggregate <target>_pae.npz) and
    # carries no provenance either.
    if r.get("n_cifs") != N_SAMPLES or r.get("n_paes") != N_SAMPLES:
        return False
    return True


def peer_done_pairs():
    """Pairs the OTHER host has already folded, from the local mirror of its progress file.

    Why a mirror and not an ssh read: the case that needs this is precisely the case where the
    peer is unreachable. When qb2 hung on 2026-07-28 its 169 records became unreadable, so
    failing its four slices over to qb1 meant either leaving half the slab stalled or refolding
    the ~125 pairs it had already completed. Neither is acceptable, and the only reason the
    choice existed is that nothing kept a local copy of the peer's pair list.

    This is a SCHEDULING input only. It never becomes a local record: the artifacts, labels and
    provenance still live on the peer, and moving them is `abag_xm_merge_hosts.py`'s job at
    release time. `abag_xm_acceptance.py` reads each host's real `progress.jsonl` directly, so a
    pair that exists only in a mirror still shows as outstanding -- the mirror can make the
    campaign skip work, never make the gate pass.
    """
    if not PEER_PROGRESS.exists():
        return set()
    seen = set()
    for line in open(PEER_PROGRESS):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if _is_done_record(r):
                seen.add((r["target"], r["model"]))
        except Exception:
            pass
    return seen


def done_pairs():
    seen = set()
    if PROGRESS.exists():
        for line in open(PROGRESS):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if not _is_done_record(r):
                    continue
                seen.add((r["target"], r["model"]))
            except Exception:
                pass
    return seen


def count_artifacts(result_dir, target):
    """The CIFs and per-sample PAEs a completed fold wrote, counted the ONE right way.

    Naming: the top-ranked sample (rank 0) is written as ``<tid>.cif`` (the "winner", no
    ``_model_0`` suffix); ranks >= 1 are ``<tid>_model_<i>.cif``. Per-sample PAEs are
    ``<tid>_model_<i>_pae.npz`` for ALL ranks including 0, plus a backwards-compat
    ``<tid>_pae.npz`` for the top-ranked sample which is deliberately EXCLUDED so the count
    equals n_samples.

    This exists as a function because it was re-implemented loosely elsewhere and the loose
    version is what produced the 8 records claiming ``n_paes: 51`` on a 50-sample campaign:
    ``glob("*_pae.npz")`` also matches the aggregate. Those records then failed the release
    preflight and could not be explained without going back to the disk. One implementation,
    one count.
    """
    struct_dir = result_dir / "structures"
    winner = struct_dir / f"{target}.cif"
    cifs = ([winner] if winner.exists() else []) + \
        sorted(struct_dir.glob(f"{target}_model_*.cif"))
    paes = sorted(struct_dir.glob(f"{target}_model_*_pae.npz"))
    return cifs, paes


def results_entry(result_dir):
    """The results.json entry for a fold, or None if it is absent or unreadable."""
    rjson = result_dir / "results.json"
    if not rjson.exists():
        return None
    try:
        results = json.load(open(rjson))
        return results[0] if isinstance(results, list) else results
    except Exception:
        return None


def write_fold_provenance(result_dir, target, tree, commit, msa_sha, mps, n_samples):
    """Write the fold's provenance beside its output, BEFORE the fold runs.

    This replaces inferring provenance after the fact, which the previous two attempts here both
    did and both got subtly wrong. A fold's engine tree is known exactly at the moment the driver
    launches it -- so write it down then, next to the artifacts, and there is nothing left to
    reconstruct if the driver dies. The orphan reconciler reads this file instead of reasoning
    about mtimes and worktree ownership.

    Why the reasoning it replaces could not be made sound: the tree lives in whichever worktree the
    DRIVER ran in, and more than one worktree can fold into the same campaign directory -- exactly
    what happened moving this campaign from p4 to p6, where p4's drivers kept folding tree
    3dc9db33 while p6 on disk was 871c8992. A directory-level "who launched this" stamp cannot
    distinguish two concurrent owners, and file mtimes cannot either. A per-fold file can, because
    the fold that wrote it is the fold it describes.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / FOLD_PROVENANCE).write_text(json.dumps(
        {"target": target, "tt_bio_tree": tree, "tt_bio_commit": commit, "msa_sha": msa_sha,
         "worktree": str(ROOT), "host": _HOST, "mps": mps, "n_samples": n_samples}) + "\n")


def read_fold_provenance(result_dir, artifacts=()):
    """The provenance the driver wrote for this fold, or None if there is none to trust.

    None means "cannot be stated", which costs a refold and never a false claim. Folds that ran
    before this file existed have none, which is correct: their tree genuinely is not recoverable.

    ``artifacts`` closes a hole found by accident: the driver writes this file and then launches the
    fold, so a fold that is interrupted *before* producing anything leaves the sidecar behind next
    to whatever a PREVIOUS run left in the same directory. Read naively, the new tree would then be
    attributed to old output. The design guarantees one ordering -- sidecar first, artifacts after --
    so an artifact older than the sidecar means the sidecar is not describing it. Observed live:
    a 21tw sidecar written 2026-07-28 13:02 beside a structures/ directory from 07-27 22:28.
    """
    f = result_dir / FOLD_PROVENANCE
    try:
        d = json.loads(f.read_text())
        written = f.stat().st_mtime
    except Exception:
        return None
    if not d.get("tt_bio_tree"):
        return None
    for art in artifacts:
        try:
            if art.stat().st_mtime < written:
                return None
        except OSError:
            return None
    return d


def _dir_bytes(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


_RESIDUES = {}


def target_residues(target):
    """Total protein residues in a target's YAML, cached. 0 if it cannot be read."""
    if target not in _RESIDUES:
        import yaml as _yaml
        try:
            d = _yaml.safe_load((YAML_DIR / f"{target}.yaml").open())
            _RESIDUES[target] = sum(
                len(v.get("sequence", ""))
                for e in d.get("sequences", []) for k, v in e.items() if k == "protein")
        except Exception:
            _RESIDUES[target] = 0
    return _RESIDUES[target]


def fold_timeout_for(target, model, cap=FOLD_TIMEOUT_S, mps=MPS, n_samples=N_SAMPLES):
    """Size-scaled timeout, clamped so it is never looser than ``cap``."""
    n = target_residues(target)
    rate = RATE_CEILING_S_PER_RES.get(model)
    if not n or rate is None:
        return cap
    if model == "boltz2" and mps and n_samples % mps:
        rate = BOLTZ2_RAGGED_CEILING_S_PER_RES
    return int(min(cap, max(FOLD_TIMEOUT_FLOOR_S, TIMEOUT_MARGIN * rate * n)))


def fold_one(target, model, device, n_samples=N_SAMPLES, mps=MPS,
             fold_timeout_s=FOLD_TIMEOUT_S, host_threads=None):
    out_dir = OUT_BASE / model.replace("-", "_")
    result_dir = out_dir / f"{RESULT_PREFIX[model]}_results_{target}"
    yaml = YAML_DIR / f"{target}.yaml"
    native = GT / f"{target}.cif"
    if not yaml.exists():
        return {"target": target, "model": model, "status": "no_yaml",
                "yaml": str(yaml)}
    # Input sanity: reject malformed YAMLs (empty sequence / missing chain)
    # BEFORE folding, so a bad YAML fails fast instead of producing silent
    # garbage (the Phase 1 regression that folded antibody-only structures for
    # 22ps/9hv9/9nw4/9udq whose antigen chain was empty).
    try:
        import yaml as _yaml
        _doc = _yaml.safe_load(yaml.read_text())
        _seqs = _doc.get("sequences", []) if _doc else []
        _bad = []
        for _s in _seqs:
            _p = _s.get("protein", {}) if isinstance(_s, dict) else {}
            _sid = _p.get("id"); _seq = _p.get("sequence", "") or ""
            if _sid in ("A", "H", "L") and len(_seq) < 5:
                _bad.append(f"{_sid}={len(_seq)}")
        if _bad or not _seqs:
            return {"target": target, "model": model, "status": "bad_yaml",
                    "yaml": str(yaml), "reason": "empty/short sequence: " + ",".join(_bad)}
    except Exception as _e:
        return {"target": target, "model": model, "status": "bad_yaml",
                "yaml": str(yaml), "reason": f"parse error: {_e}"}
    cmd = [FOLD_PY, "-m", "tt_bio.main", "predict", str(yaml),
           "--model", model, "--out_dir", str(out_dir),
           "--diffusion_samples", str(n_samples), "--max_parallel_samples", str(mps),
           "--msa_dir", str(MSA_DIR), "--msa_db_path", str(MSA_DB_PATH),
           "--seed", str(SEED), "--override", "--write_pae"]
    # One harness per card means N single-card predicts share this host. Each would
    # otherwise size its torch/OMP/BLAS pools to every core, so N folds oversubscribe
    # the CPU N-fold and the host-side work (featurization, layout conversion, output)
    # collapses -- the reason 4 cards gave 2.4x instead of 4x. Hand each fold its share.
    if host_threads:
        cmd += ["--host_threads", str(host_threads)]
    t0 = time.time()
    commit = _head_commit()
    tree = _head_tree()
    msa_sha, msa_missing = _msa_sha(target)
    # Before the fold, not after: if this driver dies mid-fold the fold keeps running and
    # completes, and this is the only thing that lets its provenance be stated afterwards.
    write_fold_provenance(result_dir, target, tree, commit, msa_sha, mps, n_samples)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "TT_VISIBLE_DEVICES": str(device),
                                 "PYTHONPATH": str(ROOT),
                                 # A card briefly stolen by the fleet dispatcher must delay
                                 # this fold, not fail it. tt_bio's default is 120 s, which is
                                 # shorter than a single fold -- so a thief that took the card
                                 # legitimately always won the race and the fold recorded
                                 # fold_failed at a constant 133 s. Wait roughly one long fold
                                 # instead; the retry loop covers anything past that.
                                 "TT_BIO_LEASE_TIMEOUT": os.environ.get(
                                     "TT_BIO_LEASE_TIMEOUT", str(LEASE_TIMEOUT_S)),
                                 "TT_BIO_LEASE_HOLDER": os.environ.get(
                                     "TT_BIO_LEASE_HOLDER",
                                     "worker:abag-xm-crossmodel-ranking-dataset-p3")},
                            start_new_session=True)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=fold_timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, _ = proc.communicate()
        rc = -9
    wall_s = time.time() - t0
    rec = {"target": target, "model": model, "wall_s": round(wall_s, 1),
           "device": device, "n_samples": n_samples, "mps": mps,
           "host": _HOST, "tt_bio_commit": commit, "tt_bio_tree": tree,
           "msa_sha": msa_sha, "msa_chains_missing": msa_missing,
           "paired_msa": False, "host_threads": host_threads,
           "timeout_s": fold_timeout_s}
    if timed_out:
        rec["status"] = "timed_out"
        rec["stderr"] = f"killed after {fold_timeout_s}s (process group); tail: {(out or '')[-1500:]}"
        return rec
    rjson = result_dir / "results.json"
    if rc < 0:
        # A negative returncode means death by signal, which is not a model failure --
        # and the child is terminated before its Rich display flushes, so `stderr` comes
        # out EMPTY. Recorded as a bare "fold_failed" that reads as an unexplained crash,
        # which is how four device-3 folds (one of them 45 min in) went undiagnosed after
        # an external SIGTERM on 2026-07-27. Name the signal so the next one is obvious.
        try:
            sig = signal.Signals(-rc).name
        except ValueError:
            sig = f"signal {-rc}"
        rec["status"] = "killed"
        rec["rc"] = rc
        rec["signal"] = sig
        rec["stderr"] = (f"killed by {sig} from outside the harness -- this campaign only "
                         f"ever sends SIGKILL, and only on the {fold_timeout_s}s timeout "
                         f"(recorded as timed_out); tail: {(out or '')[-1500:]}")
        return rec
    if rc != 0 or not rjson.exists():
        rec["status"] = "fold_failed"
        rec["rc"] = rc
        rec["stderr"] = (out or "")[-2000:]
        return rec
    try:
        results = json.load(open(rjson))
        entry = results[0] if isinstance(results, list) else results
    except Exception as e:
        rec["status"] = "bad_results_json"
        rec["stderr"] = f"{e}; tail: {(out or '')[-1000:]}"
        return rec
    struct_dir = result_dir / "structures"
    cifs, paes = count_artifacts(result_dir, target)
    # FAILED-RESULTS CROSS-CHECK: results.json status must be "ok" and the sample
    # count must match on both all_runs and the written CIF/PAE artifacts. A job
    # whose results.json says failed (or silently dropped samples) is never "ok".
    n_runs = len(entry.get("all_runs") or [])
    ok = (entry.get("status") == "ok"
          and n_runs == n_samples
          and len(cifs) == n_samples
          and len(paes) == n_samples)
    if not ok:
        rec["status"] = "incomplete"
        rec["results_status"] = entry.get("status")
        rec["n_runs"] = n_runs
        rec["n_cifs"] = len(cifs)
        rec["n_paes"] = len(paes)
        rec["stderr"] = (out or "")[-1500:]
        return rec
    rec["status"] = "ok"
    rec["confidence"] = {k: entry.get(k) for k in
                         ("confidence_score", "ptm", "iptm", "protein_iptm",
                          "complex_plddt", "runtime_s")}
    rec["n_cifs"] = len(cifs)
    rec["n_paes"] = len(paes)
    rec["bytes_structures"] = _dir_bytes(struct_dir)
    rec["result_dir"] = str(result_dir)
    rec["native_present"] = native.exists()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=None,
                    help="comma-separated subset; default = all 164 from the manifest")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=N_SAMPLES)
    ap.add_argument("--mps", type=int, default=MPS)
    ap.add_argument("--timeout", type=int, default=FOLD_TIMEOUT_S,
                    help="per-fold wall-clock timeout in seconds (default %(default)d)")
    ap.add_argument("--concurrent_folds", type=int, default=CONCURRENT_FOLDS,
                    help="how many of these harnesses run side by side on this host "
                         "(default = one per card = %(default)d). Each fold gets "
                         "cores//concurrent_folds CPU threads via predict --host_threads; "
                         "without that split every fold sizes its thread pools to all "
                         "cores and they thrash the host CPU.")
    ap.add_argument("--use_peer_mirror", action=argparse.BooleanOptionalAction, default=True,
                    help="also skip pairs the other host has already folded, read from the local "
                         "mirror of its progress file (default on). Only matters when this host "
                         "has taken over the peer's slices; without it a takeover refolds "
                         "everything the peer already did.")
    ap.add_argument("--infra_retries", type=int, default=INFRA_RETRIES,
                    help="how many times a fold that never reached the model (card held by "
                         "another process, or a chip that would not initialise) is retried "
                         "before it is recorded as a failure (default %(default)d). A "
                         "card-availability event is a scheduling condition, not a fold "
                         "result -- 37 records in the pre-p6 slab were this, not a model bug.")
    a = ap.parse_args()
    tree = provenance_or_die()
    host_threads = max(1, (os.cpu_count() or 1) // max(1, a.concurrent_folds))
    targets = a.targets.split(",") if a.targets else all_targets()
    models = a.models.split(",")
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    skip = done_pairs()
    peer = peer_done_pairs() if a.use_peer_mirror else set()
    peer_new = peer - skip
    skip |= peer
    print(f"[harness] device={a.device} targets={len(targets)} models={models} "
          f"n_samples={a.n_samples} mps={a.mps} skip={len(skip)} "
          f"host_threads={host_threads} tt_bio_tree={tree[:12]} "
          f"lease_timeout={LEASE_TIMEOUT_S}s infra_retries={a.infra_retries}"
          + (f" peer_skip={len(peer_new)}" if peer_new else ""), flush=True)
    if peer_new:
        print(f"[harness] {len(peer_new)} pair(s) skipped because the peer host already folded "
              f"them ({PEER_PROGRESS}). Their artifacts live on the peer -- "
              f"abag_xm_merge_hosts.py must run before any release table is built.", flush=True)
    dead_card = 0
    resets_used = 0
    for target in targets:
        for model in models:
            if (target, model) in skip:
                print(f"[skip] {target} {model} already ok", flush=True)
                continue
            timeout_s = fold_timeout_for(target, model, a.timeout, a.mps, a.n_samples)
            # Retry loop over infrastructure failures ONLY. A model failure (incomplete, a
            # TT_FATAL, a bad results.json) is recorded on the first attempt: retrying it
            # would burn a card re-deriving the same number and hide a real defect.
            for attempt in range(a.infra_retries + 1):
                print(f"[start] {target} {model} {time.strftime('%H:%M:%S')}"
                      + (f" (infra retry {attempt}/{a.infra_retries})" if attempt else ""),
                      flush=True)
                rec = fold_one(target, model, a.device, a.n_samples, a.mps,
                               fold_timeout_s=timeout_s, host_threads=host_threads)
                if not _is_infra_failure(rec) or attempt == a.infra_retries:
                    break
                fast = (rec.get("wall_s") or 0) < DEAD_CARD_MAX_S
                if fast and resets_used < DEAD_CARD_RESETS:
                    # A2: device-open failed outright, so the chip did not initialise. A
                    # reset is the fix and it is cheap; wait for the card otherwise.
                    resets_used += 1
                    print(f"[infra] {target} {model} failed at device-open in "
                          f"{rec.get('wall_s')}s -- resetting card {a.device} "
                          f"({resets_used}/{DEAD_CARD_RESETS} this slice)", flush=True)
                    _reset_card(a.device)
                else:
                    # A1: something else holds the card. Back off and let it finish.
                    print(f"[infra] {target} {model}: card {a.device} unavailable after "
                          f"{rec.get('wall_s')}s -- waiting {INFRA_BACKOFF_S}s", flush=True)
                    time.sleep(INFRA_BACKOFF_S)
            rec["infra_attempts"] = attempt
            with open(PROGRESS, "a") as fp:
                fp.write(json.dumps(rec) + "\n")
            print(f"[done]  {target} {model} status={rec['status']} "
                  f"wall_s={rec.get('wall_s')} n_cifs={rec.get('n_cifs')} "
                  f"n_paes={rec.get('n_paes')}", flush=True)
            # A wedged card fails every fold in ~25 s, and the driver would otherwise walk its
            # whole slice marking 60+ folds fold_failed in minutes -- observed on qb2 card 2 on
            # 2026-07-28, where a chip that needed `tt-smi -r 2` burned 9 folds before anyone
            # looked. The signature is unambiguous: the device never initialises, so the fold
            # never reaches the model, and no real fold is remotely this fast (the quickest
            # observed boltz2 fold is ~300 s). The retry loop above now resets the card and
            # tries again, so reaching here means the reset did not take: stop and leave the
            # rest of the slice untouched so a relaunch picks it up after a human looks,
            # instead of poisoning it with failures.
            if rec["status"] == "fold_failed" and (rec.get("wall_s") or 0) < DEAD_CARD_MAX_S:
                dead_card += 1
                if dead_card >= DEAD_CARD_STREAK:
                    print(f"[abort] card {a.device}: {dead_card} consecutive folds failed in "
                          f"<{DEAD_CARD_MAX_S}s and {resets_used} reset(s) did not fix it -- "
                          f"the card is not initialising and this is not a model failure. "
                          f"The rest of the slice is left untouched for a relaunch.", flush=True)
                    sys.exit(3)
            else:
                dead_card = 0
    print("CAMPAIGN SLICE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
