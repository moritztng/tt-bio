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
GT = ROOT / "examples" / "ground_truth_structures"
PROGRESS = OUT_BASE / "progress.jsonl"
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




# --- fold interpreter -------------------------------------------------------------
# NEVER use bare sys.executable here. This script's own usage line launches it as
# `PYTHONPATH=$PWD python3 scripts/abag_xm_generate.py`, i.e. SYSTEM python, which cannot
# import tt-bio's deps -- so every fold died instantly with
# `ModuleNotFoundError: No module named 'click'` while still being recorded as a
# progress.jsonl entry. That is how 270 of 438 Tier-A records became fold_failed while the
# campaign looked like it was progressing (audit 2026-07-27). The shell wrappers
# (abag_xm_tiera_watchdog.sh etc.) hardcode the venv python, which is why the folds launched
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


def done_pairs():
    seen = set()
    if PROGRESS.exists():
        for line in open(PROGRESS):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("status") != "ok":
                    continue
                if r["model"] in MPS_SENSITIVE_MODELS and r.get("mps") != MPS:
                    # Generated under a different sampling configuration -- a different
                    # procedure, so not done. The resume pass regenerates it with --override.
                    continue
                seen.add((r["target"], r["model"]))
            except Exception:
                pass
    return seen


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
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "TT_VISIBLE_DEVICES": str(device),
                                 "PYTHONPATH": str(ROOT),
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
           "host": _HOST, "tt_bio_commit": commit,
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
    # Naming: the top-ranked sample (rank 0) is written as <tid>.cif (the "winner", no
    # _model_0 suffix); ranks >= 1 are <tid>_model_<i>.cif. Per-sample PAEs are
    # <tid>_model_<i>_pae.npz for ALL ranks (incl. 0) + a backwards-compat <tid>_pae.npz
    # (top-ranked) which the glob below deliberately excludes so paes == n_samples.
    cif_winner = struct_dir / f"{target}.cif"
    cifs = ([cif_winner] if cif_winner.exists() else []) + \
        sorted(struct_dir.glob(f"{target}_model_*.cif"))
    paes = sorted(struct_dir.glob(f"{target}_model_*_pae.npz"))
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
    a = ap.parse_args()
    host_threads = max(1, (os.cpu_count() or 1) // max(1, a.concurrent_folds))
    targets = a.targets.split(",") if a.targets else all_targets()
    models = a.models.split(",")
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    skip = done_pairs()
    print(f"[harness] device={a.device} targets={len(targets)} models={models} "
          f"n_samples={a.n_samples} mps={a.mps} skip={len(skip)} "
          f"host_threads={host_threads}", flush=True)
    dead_card = 0
    for target in targets:
        for model in models:
            if (target, model) in skip:
                print(f"[skip] {target} {model} already ok", flush=True)
                continue
            print(f"[start] {target} {model} {time.strftime('%H:%M:%S')}", flush=True)
            rec = fold_one(target, model, a.device, a.n_samples, a.mps,
                           fold_timeout_s=fold_timeout_for(target, model, a.timeout,
                                                           a.mps, a.n_samples),
                           host_threads=host_threads)
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
            # observed boltz2 fold is ~300 s). Stop and leave the rest of the slice untouched so
            # a relaunch after the reset picks it up, instead of poisoning it with failures.
            if rec["status"] == "fold_failed" and (rec.get("wall_s") or 0) < DEAD_CARD_MAX_S:
                dead_card += 1
                if dead_card >= DEAD_CARD_STREAK:
                    print(f"[abort] card {a.device}: {dead_card} consecutive folds failed in "
                          f"<{DEAD_CARD_MAX_S}s -- the card is not initialising, this is not a "
                          f"model failure. Reset it (tt-smi -r {a.device}) and relaunch; the "
                          f"rest of the slice is left untouched.", flush=True)
                    sys.exit(3)
            else:
                dead_card = 0
    print("CAMPAIGN SLICE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
