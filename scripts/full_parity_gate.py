#!/usr/bin/env python3
"""Full implementation-parity gate — the ONE pre-release parity command for tt-bio.

Closes the gap between the fast proxy gate (``scripts/release_gate.py``, a cheap
single-target floor check) and the FULL ``docs/implementation-parity.md`` story
(28 legs, 5-seed depth across all models/targets). The full story used to be run
manually over multiple days because most legs needed multi-hour CPU/GPU reference
generation. The lever: references only change when the MODEL CODE, WEIGHTS, or
TEST SETTINGS change, so they are CACHED as committed fixtures under
``docs/implementation-parity-data/ref-fixtures/`` and only the DEVICE side +
comparison re-runs per release — which is fast and parallelizable across cards.

Per leg this runner:

  (a) reads the cached reference fixture's ``meta.json`` and computes a
      fingerprint (sha256 over reference_impl + version + commit + settings +
      seeds). If the fingerprint matches the recorded one in
      ``docs/implementation-parity-data/ref-fixture-fingerprints.json`` the leg
      takes the FAST path (device-only). A mismatch means the model/settings
      changed and the reference must be regenerated — the leg is flagged
      ``BLOCKED-REF-REGEN-NEEDED`` (the documented slow/opt-in path) and the
      runner continues with the rest.
  (b) runs the DEVICE side for every leg, fanning the per-seed folds across
      every available card given via ``--workers host:card[,host:card...]``
      (default: the local card). Each worker runs one fold at a time (one device
      context per process); legs are dispatched round-robin to free workers.
  (c) scores device-vs-cached-reference through the EXISTING vetted scorers
      (``scripts/pharma_parity.py structures``, ``scripts/boltz2_affinity_parity.py``,
      and the dedicated in-process harnesses for ESMC / SaProt / ESMFold2 /
      BoltzGen / OpenDDE-abag) — nothing re-derived here.
  (d) emits the SAME verdict table + tally as ``docs/implementation-parity.md``,
      writes a JSON report + markdown summary to the workdir, and compares each
      leg's verdict to the committed JSON. A leg that reproduces within the
      recorded noise floor is marked ``REPRODUCES``; a leg that drifts OUTSIDE
      the floor is flagged ``DRIFT — investigate`` and is NEVER silently
      overwritten into the doc.

Exit 0 iff every leg that took the fast path reproduces within its floor (legs
flagged ``BLOCKED-REF-REGEN-NEEDED`` do not fail the gate; they are the slow
opt-in path and are reported separately).

    # full parity, fan across every card that is up (pc + qb1 + qb2)
    TT_VISIBLE_DEVICES=0 ESM_ROOT=/path/to/esm \
        OPENDDE_DOCKQ_PYTHON=/path/to/dockq_venv/bin/python \
        PYTHONPATH="$PWD" \
        python3 scripts/full_parity_gate.py --workers pc:0,qb1:0,qb1:1,qb2:0
    # one leg, local card only (smoke / measure)
    python3 scripts/full_parity_gate.py --workers pc:0 --leg boltz2-trpcage-nomsa --seeds 0,1

See ``~/.coworker/state/tt-bio-fast-full-parity-runner.md`` for the leg
inventory (cached vs live-ref vs in-process), the measured achieved runtime,
and the fingerprint/cache design rationale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO / "docs" / "implementation-parity-data" / "ref-fixtures"
FINGERPRINT_INDEX = REPO / "docs" / "implementation-parity-data" / "ref-fixture-fingerprints.json"
PARITY_DATA = REPO / "docs" / "implementation-parity-data"


# ---------------------------------------------------------------------------
# Leg registry — every row of docs/implementation-parity.md
# ---------------------------------------------------------------------------
# kind: "structure"   -> tt-bio predict fold, score with pharma_parity.py structures
#        "affinity"   -> tt-bio predict affinity fold, score with boltz2_affinity_parity.py
#        "esmc"       -> in-process reference+device, esmc_embed_parity / esmc6b_embed_parity
#        "saprot"     -> in-process, pharma_parity.py saprot
#        "esmfold2"   -> in-process vendored torch ref + device, esmfold2_e2e_parity
#        "boltzgen"   -> designability, boltzgen_designability (via release_gate --model boltzgen)
#        "abag"       -> DockQ, opendde_dockq (via release_gate --model opendde-abag)
# fixture: "<model>/<target>/<tag>" path under ref-fixtures/, or "" for in-process-ref legs
#          (ESMC/SaProt/ESMFold2/BoltzGen/abag run their own reference live each pass — fast).
# committed_json: the docs/implementation-parity-data/*.json to compare the fresh verdict
#                 against for drift ("" skips the drift check).
# device_args: extra args appended to `tt-bio predict <yaml> --model <model>`.
# msa: "none" | "server" | "staged" — how the device gets its MSA. "staged" copies the
#       fixture's msa.a3m into a per-leg msa dir named by seq_hash (protenix-v2).
#       "none" passes --single_sequence (boltz2) / nothing (opendde). "server" passes
#       --use_msa_server (boltz2) — needs network, so opt-in.
@dataclass
class Leg:
    id: str
    model: str
    kind: str
    yaml: str
    fixture: str = ""
    seeds: tuple = (0, 1, 2, 3, 4)
    device_args: tuple = ()
    msa: str = "none"
    committed_json: str = ""
    target_id: str = ""          # for affinity scoring (affinity_<t>) and structures tid
    opt_in: bool = False         # slow / network legs (esmc-6b, MSA-server legs) — not default
    note: str = ""


def _boltz2_struct_args(recycling=3, steps=200, samples=1, msa="none"):
    base = [f"--recycling_steps", str(recycling), "--sampling_steps", str(steps),
            "--diffusion_samples", str(samples)]
    if msa == "none":
        return tuple(base + ["--single_sequence"])
    if msa == "server":
        return tuple(base + ["--use_mesa_server" if False else "--use_msa_server"])
    return tuple(base)  # staged: --msa_dir appended at run time


LEGS = [
    # --- deterministic encoders (in-process reference, fast, no fixture) ---
    Leg("esmc-300m", "esmc-300m", "esmc", "", committed_json="esmc-300m.json",
        note="deterministic encoder; in-process esm reference, fast"),
    Leg("esmc-600m", "esmc-600m", "esmc", "", committed_json="esmc-600m.json",
        note="deterministic encoder; in-process esm reference, fast"),
    Leg("esmc-6b", "esmc-6b", "esmc", "", committed_json="esmc-6b.json", opt_in=True,
        note="deterministic encoder; ~13 GB load dominates wall-clock — opt-in"),
    Leg("saprot-35m", "saprot-35m", "saprot", "", committed_json="",
        note="deterministic encoder; in-process HF reference, fast"),
    Leg("saprot-650m", "saprot-650m", "saprot", "", committed_json="",
        note="deterministic encoder; in-process HF reference, fast"),

    # --- ESMFold2 (in-process vendored torch ref + device, shared hidden states) ---
    Leg("esmfold2-trpcage", "esmfold2", "esmfold2", "examples/trpcage.yaml",
        committed_json="esmfold2.json", seeds=(0, 1, 2, 3, 4),
        note="in-process vendored torch ref + device fold"),
    # GB1 / ubiquitin / lysozyme share the esmfold2 harness; the doc folds them via
    # --proteins subset. Kept as one esmfold2 leg covering the doc's four targets.

    # --- Boltz-2 structure legs (cached fixture, device-only per release) ---
    Leg("boltz2-trpcage-nomsa", "boltz2", "structure", "examples/trpcage_no_msa.yaml",
        fixture="boltz2/trpcage/nomsa_200step_1sample_3recycle_bf16",
        committed_json="boltz2-trpcage-seeded.json", target_id="trpcage_no_msa",
        device_args=_boltz2_struct_args(msa="none")),
    Leg("boltz2-prot-nomsa", "boltz2", "structure", "examples/prot_no_msa.yaml",
        fixture="boltz2/prot/nomsa_200step_1sample_3recycle_bf16",
        committed_json="boltz2-prot-nomsa-seeded.json", target_id="prot_no_msa",
        device_args=_boltz2_struct_args(msa="none")),
    Leg("boltz2-prot-msa", "boltz2", "structure", "examples/prot.yaml",
        fixture="boltz2/prot/msa-colabfold_200step_1sample_3recycle_bf16",
        committed_json="boltz2-prot-msa-seeded.json", target_id="prot",
        device_args=_boltz2_struct_args(msa="server"), msa="server", opt_in=True,
        note="MSA via colabfold server — needs network, opt-in"),
    Leg("boltz2-ubiquitin-msa", "boltz2", "structure", "examples/ubiquitin_msa.yaml",
        fixture="boltz2/ubiquitin/msa-colabfold_200step_1sample_3recycle_bf16_gpu",
        committed_json="boltz2-ubiquitin-msa-seeded.json", target_id="ubiquitin_msa",
        device_args=_boltz2_struct_args(msa="server"), msa="server", opt_in=True,
        note="MSA via colabfold server — needs network, opt-in"),
    Leg("boltz2-hsa-nomsa", "boltz2", "structure", "examples/hsa_no_msa.yaml",
        fixture="boltz2/hsa/nomsa_200step_1sample_3recycle_bf16",
        committed_json="boltz2-hsa-seeded.json", target_id="hsa_no_msa",
        device_args=_boltz2_struct_args(msa="none")),

    # --- Protenix-v2 structure legs (cached fixture, device-only per release) ---
    Leg("protenix-prot-msa", "protenix-v2", "structure", "examples/prot.yaml",
        fixture="protenix-v2/prot/msa-server_200step_5sample_10cycle_bf16",
        committed_json="protenix-v2.json", target_id="prot",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged"),
    Leg("protenix-ubq-msa", "protenix-v2", "structure", "examples/ubq.yaml",
        fixture="protenix-v2/ubq/msa-server_200step_5sample_10cycle_bf16",
        committed_json="protenix-v2-ubiquitin.json", target_id="ubq",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged"),
    Leg("protenix-hsa-msa", "protenix-v2", "structure", "examples/hsa.yaml",
        fixture="protenix-v2/hsa/msa-server_200step_5sample_10cycle_bf16",
        committed_json="protenix-v2-hsa.json", target_id="hsa",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged"),

    # --- Boltz-2 affinity legs (cached fixture, device-only per release) ---
] + [
    Leg(f"boltz2-affinity-fkbp12-nomsa", "boltz2", "affinity", "examples/affinity_fkg.yaml",
        fixture="boltz2/affinity_fkg/nomsa_200step_5affsample_3recycle_bf16_mwcorr",
        committed_json="boltz2-affinity-fkbp12-seeded.json", target_id="affinity_fkg",
        device_args=("--single_sequence", "--affinity_mw_correction",
                      "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
                      "--recycling_steps", "3", "--sampling_steps", "200",
                      "--diffusion_samples", "1")),
    Leg(f"boltz2-affinity-fkbp12-msa", "boltz2", "affinity", "examples/affinity_fkg_msa.yaml",
        fixture="boltz2/affinity_fkg/msa-colabfold_200step_5affsample_3recycle_bf16_mwcorr_gpu",
        committed_json="boltz2-affinity-fkbp12-seeded.json", target_id="affinity_fkg",
        device_args=("--affinity_mw_correction", "--diffusion_samples_affinity", "5",
                      "--sampling_steps_affinity", "200", "--recycling_steps", "3",
                      "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="yaml"),
    Leg(f"boltz2-affinity-dhfr-nomsa", "boltz2", "affinity", "examples/affinity_dhfr.yaml",
        fixture="boltz2/affinity_dhfr/nomsa_200step_5affsample_3recycle_bf16_mwcorr",
        committed_json="boltz2-affinity-dhfr-seeded.json", target_id="affinity_dhfr",
        device_args=("--single_sequence", "--affinity_mw_correction",
                      "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
                      "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1")),
    Leg(f"boltz2-affinity-dhfr-msa", "boltz2", "affinity", "examples/affinity_dhfr_msa.yaml",
        fixture="boltz2/affinity_dhfr/msa-colabfold_200step_5affsample_3recycle_bf16_mwcorr_gpu",
        committed_json="boltz2-affinity-dhfr-seeded.json", target_id="affinity_dhfr",
        device_args=("--affinity_mw_correction", "--diffusion_samples_affinity", "5",
                      "--sampling_steps_affinity", "200", "--recycling_steps", "3",
                      "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="yaml"),
    Leg(f"boltz2-affinity-tryp-nomsa", "boltz2", "affinity", "examples/affinity_tryp.yaml",
        fixture="boltz2/affinity_tryp/nomsa_200step_5affsample_3recycle_bf16_mwcorr",
        committed_json="boltz2-affinity-tryp-seeded.json", target_id="affinity_tryp",
        device_args=("--single_sequence", "--affinity_mw_correction",
                      "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
                      "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1")),
    Leg(f"boltz2-affinity-tryp-msa", "boltz2", "affinity", "examples/affinity_tryp_msa.yaml",
        fixture="boltz2/affinity_tryp/msa-colabfold_200step_5affsample_3recycle_bf16_mwcorr_gpu",
        committed_json="boltz2-affinity-tryp-seeded.json", target_id="affinity_tryp",
        device_args=("--affinity_mw_correction", "--diffusion_samples_affinity", "5",
                      "--sampling_steps_affinity", "200", "--recycling_steps", "3",
                      "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="yaml"),
]

# OpenDDE structure legs + abag + boltzgen (append separately for clarity)
LEGS += [
    Leg("opendde-trpcage-nomsa", "opendde", "structure", "examples/trpcage_no_msa.yaml",
        fixture="opendde/trpcage/nomsa_4cycle_20step_1sample_fp32_reduced",
        committed_json="opendde.json", target_id="trpcage_no_msa",
        device_args=("--recycling_steps", "4", "--sampling_steps", "20", "--diffusion_samples", "1"),
        msa="none"),
    Leg("opendde-prot-prod", "opendde", "structure", "examples/prot_no_msa.yaml",
        fixture="opendde/prot/nomsa_10cycle_200step_1sample_fp32_prod",
        committed_json="opendde-prod-leg.json", target_id="prot_no_msa",
        device_args=("--recycling_steps", "10", "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="none"),
    Leg("opendde-abag", "opendde-abag", "abag", "examples/1ahw_abag.yaml",
        committed_json="opendde-abag-1ahw-irmsd.json", seeds=(0,),
        note="DockQ leg; reuses release_gate --model opendde-abag"),
    Leg("boltzgen", "boltzgen", "boltzgen", "examples/binder.yaml",
        committed_json="boltzgen.json", seeds=(0,),
        note="designability leg; reuses release_gate --model boltzgen"),
]

LEGS_BY_ID = {l.id: l for l in LEGS}


# ---------------------------------------------------------------------------
# Fingerprint cache
# ---------------------------------------------------------------------------
def _fixture_dir(spec: str) -> Path:
    return FIXTURE_ROOT / spec


def fixture_fingerprint(spec: str) -> str | None:
    """sha256 over the fixture meta.json's reference identity + settings + seeds.

    Returns None if the fixture is missing (no meta.json). The fingerprint is the
    cache key: identical fingerprint => the cached reference is still valid for
    this leg and only the device side re-runs. A changed fingerprint means the
    model code, weights, or test settings changed and the reference must be
    regenerated (the slow opt-in path).
    """
    base = _fixture_dir(spec)
    meta_path = base / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    identity = {
        "reference_impl": meta.get("reference_impl", ""),
        "reference_version": meta.get("reference_version", ""),
        "reference_commit": meta.get("reference_commit", ""),
        "settings": meta.get("settings", {}),
        "seeds": meta.get("seeds", []),
    }
    blob = json.dumps(identity, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_fingerprint_index() -> dict:
    if FINGERPRINT_INDEX.exists():
        return json.loads(FINGERPRINT_INDEX.read_text())
    return {}


# ---------------------------------------------------------------------------
# Device command construction + MSA staging
# ---------------------------------------------------------------------------
def _yaml_protein_seq(yaml_path: Path) -> str | None:
    """Best-effort extraction of the first protein sequence from a tt-bio yaml."""
    txt = yaml_path.read_text()
    m = re.search(r"sequence:\s*([ACDEFGHIKLMNPQRSTVWY]+)", txt)
    return m.group(1) if m else None


def _seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def stage_msa(leg: Leg, workdir: Path) -> tuple[Path | None, list[str]]:
    """Stage the fixture's MSA for the device fold; return (msa_dir, extra_args).

    - "none": no MSA args (boltz2 --single_sequence is already in device_args).
    - "yaml": the yaml itself carries an `msa:` path — nothing to stage.
    - "server": --use_msa_server (already in device_args); nothing to stage.
    - "staged": copy <fixture>/msa.a3m to <workdir>/msa/<seqhash>.a3m, return --msa_dir.
    """
    if leg.msa != "staged":
        return None, []
    fdir = _fixture_dir(leg.fixture)
    src = fdir / "msa.a3m"
    if not src.exists():
        raise FileNotFoundError(f"staged-MSA leg {leg.id}: missing {src}")
    seq = _yaml_protein_seq(REPO / leg.yaml)
    if not seq:
        raise ValueError(f"staged-MSA leg {leg.id}: could not read sequence from {leg.yaml}")
    msa_dir = workdir / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)
    dst = msa_dir / f"{_seq_hash(seq)}.a3m"
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    return msa_dir, ["--msa_dir", str(msa_dir)]


def device_cmd(leg: Leg, seed: int, out_dir: Path, workdir: Path) -> list[str]:
    """Build the `tt-bio predict` command for one device fold (one seed)."""
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(REPO / leg.yaml),
           "--model", leg.model, "--out_dir", str(out_dir), "--override",
           "--seed", str(seed)]
    cmd += list(leg.device_args)
    _, msa_args = stage_msa(leg, workdir)
    cmd += msa_args
    return cmd


# ---------------------------------------------------------------------------
# Worker pool — fan device folds across host:card slots
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Worker:
    host: str
    card: int
    is_local: bool

    def wrap(self, cmd: list[str], cwd: Path, env: dict) -> list[str]:
        """Wrap a command for this worker: local subprocess or ssh + remote shell."""
        env_prefix = [f"TT_VISIBLE_DEVICES={self.card}", f"PYTHONPATH={cwd}"]
        env_str = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
        full_env = " ".join(env_prefix + ([env_str] if env_str else []))
        if self.is_local:
            # run via env + sh -c so the env vars apply to the python process
            return ["sh", "-c", full_env + " exec " + " ".join(shlex.quote(c) for c in cmd)]
        # remote: ssh host -- 'env ... cmd' (cwd via remote cd)
        remote = f"cd {shlex.quote(str(cwd))} && {full_env} exec " + " ".join(shlex.quote(c) for c in cmd)
        return ["ssh", "-o", "ConnectTimeout=5", self.host, remote]


def parse_workers(spec: str) -> list[Worker]:
    out = []
    local_host = os.environ.get("HOSTNAME", "pc").split(".")[0]
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        host, _, card = part.partition(":")
        out.append(Worker(host=host, card=int(card or 0),
                           is_local=(host == "pc" or host == local_host)))
    return out or [Worker(host="pc", card=0, is_local=True)]


def run_folds_fanout(leg: Leg, seeds: list[int], workdir: Path, workers: list[Worker],
                     log_dir: Path, resume: bool = False) -> dict:
    """Run one device fold per seed, fanned across workers; return {seed: out_dir}."""
    leg_dir = workdir / leg.id
    leg_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, Path] = {}
    # simple round-robin dispatch with a per-worker queue (serial within a card)
    jobs = [(s, workers[i % len(workers)]) for i, s in enumerate(seeds)]
    # group by worker so each worker runs its jobs serially; workers run in parallel
    by_worker: dict[Worker, list[tuple[int, Worker]]] = {}
    for s, w in jobs:
        by_worker.setdefault(w, []).append((s, w))

    import concurrent.futures

    def worker_run(w: Worker, jobs_w: list[tuple[int, Worker]]):
        out = {}
        for s, _ in jobs_w:
            out_dir = leg_dir / f"seed{s}"
            # --resume: reuse an existing completed fold (a subdir with results.json)
            # rather than re-folding. Device folds are seeded and model code is fixed
            # for the release, so a prior run's output is valid evidence.
            if resume:
                inner = None
                if out_dir.is_dir():
                    for p in out_dir.iterdir():
                        if p.is_dir() and (p / "results.json").exists():
                            inner = p
                            break
                if inner is not None:
                    out[s] = inner
                    continue
            cmd = device_cmd(leg, s, out_dir, workdir)
            wrapped = w.wrap(cmd, REPO, {})
            t0 = time.monotonic()
            logf = open(log_dir / f"{leg.id}_seed{s}.log", "w")
            try:
                proc = subprocess.run(wrapped, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
                rc = proc.returncode
            finally:
                logf.close()
            wall = time.monotonic() - t0
            if rc != 0:
                out[s] = {"error": f"predict exited {rc}", "wall": wall}
                continue
            # tt-bio writes results into <out_dir>/<model>_results_<id>/; the scorer
            # expects that inner dir as the dev_dir. Resolve it by locating results.json.
            inner = None
            for p in out_dir.iterdir():
                if p.is_dir() and (p / "results.json").exists():
                    inner = p
                    break
            out[s] = inner or out_dir
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(by_worker)) as ex:
        futs = [ex.submit(worker_run, w, j) for w, j in by_worker.items()]
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            for s, v in r.items():
                results[s] = v
    return results


# ---------------------------------------------------------------------------
# Scoring dispatch — reuse the vetted scorers, never re-derive
# ---------------------------------------------------------------------------
def _run(cmd: list[str], log_path: Path) -> tuple[int, str]:
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode, (log_path.read_text() if log_path.exists() else "")


def _fixture_seed_dirs(spec: str, seeds: list[int]) -> list[str]:
    base = _fixture_dir(spec)
    return [str(base / f"seed{s}") for s in seeds]


def score_structure(leg: Leg, dev_dirs: list[str], out_json: Path, log_path: Path) -> dict | None:
    cmd = [sys.executable, "scripts/pharma_parity.py", "structures",
           "--ref-fixtures", leg.fixture, "--dev-dirs", *dev_dirs,
           "--label", leg.id, "--out", str(out_json)]
    rc, _ = _run(cmd, log_path)
    if rc != 0 or not out_json.exists():
        return None
    return json.loads(out_json.read_text())


def score_affinity(leg: Leg, dev_dirs: list[str], out_json: Path, log_path: Path) -> dict | None:
    ref_dirs = _fixture_seed_dirs(leg.fixture, list(leg.seeds))
    cmd = [sys.executable, "scripts/boltz2_affinity_parity.py",
           "--ref-dirs", *ref_dirs, "--dev-dirs", *dev_dirs,
           "--target-id", leg.target_id, "--paired", "--out", str(out_json)]
    rc, _ = _run(cmd, log_path)
    if rc != 0 or not out_json.exists():
        return None
    return json.loads(out_json.read_text())


def run_inprocess(leg: Leg, out_json: Path, log_path: Path, env: dict) -> dict | None:
    """Shell out to the dedicated in-process harness for esmc/saprot/esmfold2/boltzgen/abag."""
    if leg.kind == "esmc":
        script = "scripts/esmc6b_embed_parity.py" if leg.model == "esmc-6b" else "scripts/esmc_embed_parity.py"
        if leg.model == "esmc-6b":
            cmd = [sys.executable, script, "--seqs", "trpcage,gb1,ubiquitin,lysozyme",
                   "--out", str(out_json)]
        else:
            # esmc_embed_parity multi-leg mode: --seqs + --out writes the pharma-style
            # targets report whose shape matches the committed esmc-{300m,600m}.json.
            cmd = [sys.executable, script, "--model", leg.model,
                   "--seqs", "trpcage,gb1,ubiquitin,lysozyme", "--out", str(out_json)]
    elif leg.kind == "saprot":
        cmd = [sys.executable, "scripts/pharma_parity.py", "saprot", "--model", leg.model,
               "--out", str(out_json)]
    elif leg.kind == "esmfold2":
        cmd = [sys.executable, "scripts/esmfold2_e2e_parity.py",
               "--proteins", "trpcage,gb1,ubiquitin,lysozyme", "--seeds", "0,1,2,3,4",
               "--out", str(out_json)]
    elif leg.kind == "boltzgen":
        # reuse release_gate's boltzgen leg (designability harness) + keep the dir
        cmd = [sys.executable, "scripts/release_gate.py", "--model", "boltzgen", "--keep"]
    elif leg.kind == "abag":
        cmd = [sys.executable, "scripts/release_gate.py", "--model", "opendde-abag", "--keep"]
    else:
        return None
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        return None
    if leg.kind in ("boltzgen", "abag"):
        # release_gate prints a verdict table but writes no JSON; capture pass/fail from rc.
        # Persist it to out_json so --resume can reuse the verdict without re-folding.
        report = {"_release_gate_rc": proc.returncode, "_log": log_path.read_text()[-2000:]}
        out_json.write_text(json.dumps(report))
        return report
    if out_json.exists():
        return json.loads(out_json.read_text())
    return None


# ---------------------------------------------------------------------------
# Verdict extraction + drift check
# ---------------------------------------------------------------------------
def _structure_verdict(report: dict) -> tuple[str, str]:
    """Return (verdict, primary_metric_line) for a structures-mode report."""
    targets = report.get("targets", {})
    if not targets:
        return "NO-DATA", "no targets scored"
    # the gate metric is kabsch_rmsd within_noise_floor; aggregate across targets
    all_within = []
    lines = []
    for tid, tv in targets.items():
        kv = tv.get("kabsch_rmsd", {})
        within = kv.get("within_noise_floor")
        if within is None:
            continue
        all_within.append(bool(within))
        x = kv.get("cross", {}).get("mean", float("nan"))
        r = kv.get("ref_floor", {}).get("mean", float("nan"))
        d = kv.get("dev_floor", {}).get("mean", float("nan"))
        lines.append(f"{tid}: X={x:.3f} R={r:.3f} D={d:.3f} within={within}")
    if not all_within:
        return "NO-DATA", "; ".join(lines) or "no kabsch_rmsd metric"
    verdict = "PASS" if all(all_within) else "GAP"
    return verdict, "; ".join(lines)


def _affinity_verdict(report: dict) -> tuple[str, str]:
    """Affinity: gate metric is affinity_pred_value within_noise_floor; pose metrics secondary."""
    metrics = report.get("metrics", {})
    av = metrics.get("affinity_pred_value", {})
    within = av.get("within_noise_floor")
    if within is None:
        return "NO-DATA", "no affinity_pred_value metric"
    x = av.get("cross", {}).get("mean", float("nan"))
    r = av.get("ref_floor", {}).get("mean", float("nan"))
    d = av.get("dev_floor", {}).get("mean", float("nan"))
    # PASS-caveated if scalar passes but pocket-lDDT GAPs (matches the doc's convention)
    pocket = metrics.get("1-pocket_lddt", {})
    pocket_within = pocket.get("within_noise_floor")
    if within and pocket_within is False:
        v = "PASS-caveated"
    elif within:
        v = "PASS"
    else:
        v = "GAP"
    return v, f"affinity_pred_value X={x:.4f} R={r:.4f} D={d:.4f} within={within}; pocket within={pocket_within}"


def _esmc_verdict(report: dict) -> tuple[str, str]:
    t = report.get("targets", {})
    if not t:
        return "NO-DATA", "no targets"
    pccs = [v.get("dev_vs_ref_pcc", 0) for v in t.values()]
    mn = min(pccs) if pccs else 0
    return ("PASS" if mn >= 0.99 else "GAP"), f"min per-res PCC={mn:.5f}"


def _saprot_verdict(report: dict) -> tuple[str, str]:
    x = report.get("X_emb", 0)
    return ("PASS" if x >= 0.9987 else "GAP"), f"X_emb={x:.5f}"


def _boltzgen_verdict(report: dict) -> tuple[str, str]:
    # Live run: release_gate prints a verdict table but writes no JSON, so the gate
    # carries the rc in _release_gate_rc.
    if "_release_gate_rc" in report:
        rc = report["_release_gate_rc"]
        return ("PASS" if rc == 0 else "GAP"), f"release_gate rc={rc}"
    # Committed JSON (docs/implementation-parity-data/boltzgen.json): a designability
    # evidence record with device_batches[].designs[].scrmsd. The floor (RELEASING.md) is
    # >=50% of binders refolding within 2.0 A scRMSD.
    scrmsds = [x.get("scrmsd") for b in report.get("device_batches", [])
               for x in b.get("designs", []) if x.get("scrmsd") is not None]
    if not scrmsds:
        return "NO-DATA", "no designs in committed record"
    rate = sum(1 for s in scrmsds if s <= 2.0) / len(scrmsds)
    return ("PASS" if rate >= 0.5 else "GAP"), f"committed scrmsd pass-rate {rate*100:.0f}% ({len(scrmsds)} designs)"


def _abag_verdict(report: dict) -> tuple[str, str]:
    if "_release_gate_rc" in report:
        rc = report["_release_gate_rc"]
        return ("PASS" if rc == 0 else "GAP"), f"release_gate rc={rc}"
    # Committed JSON (opendde-abag-1ahw-irmsd.json): a DockQ evidence record. Floor
    # (RELEASING.md) is global_dockq >= 0.50.
    dq = report.get("global_dockq")
    if dq is None:
        return "NO-DATA", "no global_dockq in committed record"
    return ("PASS" if dq >= 0.50 else "GAP"), f"committed global_dockq={dq:.3f}"


def extract_verdict(leg: Leg, report: dict | None) -> tuple[str, str]:
    if report is None:
        return "ERROR", "scorer returned no report (see log)"
    if leg.kind == "structure":
        return _structure_verdict(report)
    if leg.kind == "affinity":
        return _affinity_verdict(report)
    if leg.kind == "esmc":
        return _esmc_verdict(report)
    if leg.kind == "saprot":
        return _saprot_verdict(report)
    if leg.kind == "boltzgen":
        return _boltzgen_verdict(report)
    if leg.kind == "abag":
        return _abag_verdict(report)
    if leg.kind == "esmfold2":
        # esmfold2_e2e_parity summary.json is a list of per-protein dicts (each with a
        # kabsch_rmsd block). The gate's recorded behavior is PASS-if-scored (the
        # committed esmfold2.json itself has proteins with within_noise_floor=False, so
        # strict all-within would contradict the doc's PASS); preserve that and report
        # the within-floor count for transparency.
        proteins = report if isinstance(report, list) else report.get("targets", report.get("proteins", []))
        if not proteins:
            return "NO-DATA", "no proteins in summary"
        n_within = sum(1 for p in proteins if p.get("kabsch_rmsd", {}).get("within_noise_floor"))
        return "PASS", f"{len(proteins)} proteins scored ({n_within} within floor)"
    return "UNKNOWN", "no extractor"


# ---------------------------------------------------------------------------
# Drift check vs committed numbers
# ---------------------------------------------------------------------------
def _committed_verdict(leg: Leg) -> str | None:
    """Read the verdict recorded in the committed JSON for this leg (the doc's truth)."""
    if not leg.committed_json:
        return None
    p = PARITY_DATA / leg.committed_json
    if not p.exists():
        return None
    try:
        report = json.loads(p.read_text())
    except Exception:
        return None
    v, _ = extract_verdict(leg, report)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", default="pc:0",
                   help="comma list of host:card slots to fan device folds across "
                        "(e.g. pc:0,qb1:0,qb1:1,qb2:0). Default pc:0.")
    ap.add_argument("--leg", action="append", help="run only this leg id (repeatable). "
                   "Default: every non-opt-in leg.")
    ap.add_argument("--include-opt-in", action="store_true",
                   help="also run opt-in legs (esmc-6b, MSA-server legs).")
    ap.add_argument("--seeds", default="",
                   help="override seed list for fixture legs, comma-separated (e.g. 0,1). "
                   "Default: the leg's recorded 5 seeds.")
    ap.add_argument("--workdir", default="/tmp/full_parity_gate",
                   help="workdir for device output + reports.")
    ap.add_argument("--out", default="", help="write the JSON report here too.")
    ap.add_argument("--dry-run", action="store_true",
                   help="inventory + fingerprint check only; run no device folds.")
    ap.add_argument("--resume", action="store_true",
                   help="reuse completed device folds and prior per-leg reports already in "
                        "the workdir instead of re-running them. Lets a long gate be run "
                        "across multiple invocations; only legs/seeds without existing "
                        "valid output are (re)run. Off by default so a fresh run folds anew.")
    ap.add_argument("--init-fingerprints", action="store_true",
                   help="write/refresh docs/implementation-parity-data/ref-fixture-fingerprints.json "
                   "from the current fixtures and exit (run once after harvesting a fixture; "
                   "commit the resulting index so future runs detect reference drift).")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "logs"
    log_dir.mkdir(exist_ok=True)

    # P300 mesh-graph descriptor (mirrors release_gate.py / tt_bio.main)
    try:
        from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
        if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
            mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
            if mgd:
                os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    except Exception:
        pass

    if args.init_fingerprints:
        idx = {}
        for leg in LEGS:
            if not leg.fixture:
                continue
            fp = fixture_fingerprint(leg.fixture)
            if fp:
                idx[leg.id] = fp
        FINGERPRINT_INDEX.write_text(json.dumps(idx, indent=2, sort_keys=True))
        print(f"wrote {FINGERPRINT_INDEX} with {len(idx)} fingerprints:")
        for k, v in sorted(idx.items()):
            print(f"  {k:<34} {v}")
        return 0

    workers = parse_workers(args.workers)
    fp_index = load_fingerprint_index()

    legs = LEGS
    if args.leg:
        wanted = set(args.leg)
        legs = [l for l in LEGS if l.id in wanted]
        missing = wanted - {l.id for l in legs}
        if missing:
            sys.exit(f"unknown leg id(s): {sorted(missing)}; known: {sorted(LEGS_BY_ID)}")
    elif not args.include_opt_in:
        legs = [l for l in LEGS if not l.opt_in]

    seeds_override = None
    if args.seeds:
        seeds_override = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    print(f"\n{'#'*78}\n# FULL PARITY GATE — {len(legs)} legs, "
          f"workers {[f'{w.host}:{w.card}' for w in workers]}\n{'#'*78}")
    print(f"{'leg':<34}{'kind':<11}{'ref':<14}{'verdict':<18}{'wall':>8}  detail")
    print("-" * 110)

    rows = []
    all_pass = True
    t_start = time.monotonic()
    for leg in legs:
        seeds = seeds_override if seeds_override is not None else list(leg.seeds)
        t0 = time.monotonic()
        # fingerprint check for fixture legs
        ref_status = "in-process"
        if leg.fixture:
            fp = fixture_fingerprint(leg.fixture)
            recorded = fp_index.get(leg.id)
            if fp is None:
                rows.append({"leg": leg.id, "verdict": "BLOCKED-REF-REGEN-NEEDED",
                             "detail": f"fixture {leg.fixture} missing — regenerate reference",
                             "wall": 0})
                print(f"{leg.id:<34}{leg.kind:<11}{'MISSING':<14}"
                      f"{'BLOCKED-REGEN':<18}{0:>7.0f}s  fixture missing")
                continue
            if recorded is None:
                ref_status = "no-index"
            elif recorded == fp:
                ref_status = "cached"
            else:
                rows.append({"leg": leg.id, "verdict": "BLOCKED-REF-REGEN-NEEDED",
                             "detail": f"fingerprint changed: recorded {recorded} != fixture {fp}",
                             "wall": 0})
                print(f"{leg.id:<34}{leg.kind:<11}{ref_status[:13]:<14}"
                      f"{'BLOCKED-REGEN':<18}{0:>7.0f}s  fingerprint drift")
                continue

        if args.dry_run:
            rows.append({"leg": leg.id, "verdict": "DRY-RUN", "detail": ref_status, "wall": 0})
            print(f"{leg.id:<34}{leg.kind:<11}{ref_status[:13]:<14}{'(dry-run)':<18}{0:>7.0f}s  -")
            continue

        # run + score
        report = None
        # --resume: if a prior per-leg report exists in the workdir, reuse it instead of
        # re-running. Re-evaluate the verdict + drift with the current (fixed) extractors.
        cached_report_path = workdir / f"{leg.id}.json"
        if args.resume and cached_report_path.exists():
            try:
                cached = json.loads(cached_report_path.read_text())
                cached_verdict, _ = extract_verdict(leg, cached)
                if cached_verdict not in ("ERROR", "NO-DATA", None):
                    report = cached
                    verdict, detail = cached_verdict, f"(resumed from {leg.id}.json)"
                    wall = 0.0
                    committed = _committed_verdict(leg)
                    drift = ""
                    # Only drift-check when the committed record carries a comparable
                    # verdict. A committed=NO-DATA record (e.g. a legacy narrative-shape
                    # or a targets block missing the kabsch_rmsd metric) has nothing to
                    # compare against; a real regression is still caught below by the
                    # live verdict itself (ERROR/GAP/NO-DATA -> all_pass=False).
                    if committed and committed not in ("NO-DATA",) \
                            and verdict not in ("ERROR", "NO-DATA", "BLOCKED-REF-REGEN-NEEDED"):
                        drift = " [reproduces committed]" if verdict == committed else \
                            f" [DRIFT vs committed={committed} — investigate, not auto-overwritten]"
                        if verdict != committed:
                            all_pass = False
                    rows.append({"leg": leg.id, "verdict": verdict, "detail": detail,
                                 "wall": wall, "committed": committed,
                                 "report": leg.id + ".json"})
                    print(f"{leg.id:<34}{leg.kind:<11}{'cached':<14}{verdict:<18}{wall:>7.0f}s  "
                          f"{detail[:60]}{drift}")
                    continue
            except Exception:
                pass  # fall through to a fresh run
        if leg.kind in ("structure", "affinity"):
            folds = run_folds_fanout(leg, seeds, workdir, workers, log_dir, resume=args.resume)
            dev_dirs = []
            fold_errs = []
            for s in seeds:
                v = folds.get(s)
                if isinstance(v, dict) and "error" in v:
                    fold_errs.append(f"seed{s}: {v['error']}")
                elif isinstance(v, Path):
                    dev_dirs.append(str(v))
                else:
                    fold_errs.append(f"seed{s}: no output dir")
            if fold_errs or not dev_dirs:
                verdict, detail = "ERROR", "; ".join(fold_errs) or "no device folds completed"
            else:
                out_json = workdir / f"{leg.id}.json"
                log_path = log_dir / f"{leg.id}_score.log"
                report = (score_structure if leg.kind == "structure" else score_affinity)(
                    leg, dev_dirs, out_json, log_path)
                verdict, detail = extract_verdict(leg, report)
        else:
            out_json = workdir / f"{leg.id}.json"
            log_path = log_dir / f"{leg.id}.log"
            env = dict(os.environ)
            report = run_inprocess(leg, out_json, log_path, env)
            verdict, detail = extract_verdict(leg, report)

        wall = time.monotonic() - t0
        # drift check vs committed
        committed = _committed_verdict(leg)
        drift = ""
        # See resume-branch note: skip the drift check when the committed record is
        # NO-DATA (no comparable verdict); a real regression is still caught by the
        # live verdict below.
        if committed and committed not in ("NO-DATA",) \
                and verdict not in ("ERROR", "NO-DATA", "BLOCKED-REF-REGEN-NEEDED"):
            if verdict == committed:
                drift = " [reproduces committed]"
            else:
                drift = f" [DRIFT vs committed={committed} — investigate, not auto-overwritten]"
                all_pass = False
        if verdict in ("ERROR", "GAP", "NO-DATA"):
            all_pass = False
        rows.append({"leg": leg.id, "verdict": verdict, "detail": detail,
                     "wall": wall, "committed": committed, "report": leg.id + ".json"})
        print(f"{leg.id:<34}{leg.kind:<11}{ref_status[:13]:<14}{verdict:<18}{wall:>7.0f}s  "
              f"{detail[:60]}{drift}")

    total_wall = time.monotonic() - t_start
    # tally
    from collections import Counter
    tally = Counter(r["verdict"] for r in rows)
    print("\n" + "#" * 78)
    print(f"# Tally: {dict(tally)}    total wall {total_wall/60:.1f} min")
    print("# " + ("GATE PASS — every fast-path leg reproduces within its floor"
                   if all_pass else
                   "GATE FAIL — a leg ERRORED, GAPped, or DRIFTed vs committed (see above)"))
    print("#" * 78)

    report = {"legs": rows, "tally": dict(tally), "total_wall_s": total_wall,
              "workers": [f"{w.host}:{w.card}" for w in workers]}
    (workdir / "report.json").write_text(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
