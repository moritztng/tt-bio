"""Job engine for the ai& Bio platform.

A :class:`JobManager` owns a workspace directory and a single background worker
that runs queued jobs one at a time (serialised so concurrent runs never fight
over the same Tenstorrent devices). Each job shells out to the real ``tt-bio``
CLI — ``tt-bio predict`` for structure/affinity prediction and ``tt-bio gen
run`` for BoltzGen design — so results are identical to running tt-bio by hand.

Everything here is framework-agnostic; the Flask app in ``app.py`` is a thin
HTTP skin over this.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from queue import Queue
from typing import Any

from . import limits

# ---------------------------------------------------------------------------
# Resolving the tt-bio entry point
# ---------------------------------------------------------------------------

def _resolve_ttbio() -> list[str]:
    """Find how to invoke tt-bio. Prefer the console script next to the current
    interpreter; fall back to ``python -m tt_bio.main``."""
    console = Path(sys.executable).with_name("tt-bio")
    if console.exists():
        return [str(console)]
    return [sys.executable, "-m", "tt_bio.main"]


TTBIO = _resolve_ttbio()

# Job lifecycle states.
QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELED = (
    "queued", "running", "succeeded", "failed", "canceled",
)

# Coarse pipeline stages we look for in BoltzGen logs, in order.
_DESIGN_STAGES = ["design", "inverse_folding", "folding", "analysis", "filtering"]


@dataclasses.dataclass
class Job:
    id: str
    kind: str                      # "predict" | "design"
    name: str
    created_at: float
    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    model: str | None = None       # predict
    protocol: str | None = None    # design
    status: str = QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    error: str | None = None
    total: int = 0                 # number of targets (predict)
    done: int = 0
    stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["progress"] = self.progress()
        return d

    def progress(self) -> float | None:
        if self.status in (SUCCEEDED, CANCELED):
            return 1.0
        if self.kind == "predict" and self.total:
            return min(0.99, self.done / self.total) if self.status == RUNNING else 0.0
        if self.kind == "design" and self.stage in _DESIGN_STAGES:
            return min(0.99, (_DESIGN_STAGES.index(self.stage) + 0.5) / len(_DESIGN_STAGES))
        return None  # indeterminate


class JobManager:
    def __init__(self, workspace: str | Path, *, cluster=None, max_concurrent: int = 32):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.RLock()
        self.queue: "Queue[str]" = Queue()
        # When a cluster is attached, predict jobs submit to its shared
        # controller and use no local device directly, so many run at once.
        # Design jobs (and predicts with no cluster) need the local devices, so
        # they run exclusively. The scheduler below enforces that split.
        self.cluster = cluster
        self.max_concurrent = max(1, int(max_concurrent))
        self._sched = threading.Condition()
        self._running = 0          # jobs currently executing (any kind)
        self._excl_active = False  # an exclusive (device-owning) job is running
        self._excl_waiting = 0     # exclusive jobs waiting to start
        self._load_existing()
        self._pool = [threading.Thread(target=self._run_loop, daemon=True)
                      for _ in range(self.max_concurrent)]
        for t in self._pool:
            t.start()

    # -- scheduler gating --------------------------------------------------
    def _admit(self, exclusive: bool) -> None:
        """Block until this job may run. Concurrent (controller-predict) jobs
        wait only while an exclusive job is active or queued; exclusive jobs
        (design, or predict with no cluster) wait for every running job to
        drain, then run alone."""
        with self._sched:
            if exclusive:
                self._excl_waiting += 1
                self._sched.notify_all()  # stop admitting new concurrent jobs
                while self._running > 0 or self._excl_active:
                    self._sched.wait()
                self._excl_waiting -= 1
                self._excl_active = True
            else:
                while self._excl_active or self._excl_waiting > 0:
                    self._sched.wait()
            self._running += 1

    def _retire(self, exclusive: bool) -> None:
        with self._sched:
            self._running = max(0, self._running - 1)
            if exclusive:
                self._excl_active = False
            self._sched.notify_all()

    # -- directories -------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.workspace / job_id

    def _inputs_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "inputs"

    def _out_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "out"

    def _log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "run.log"

    def _meta_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "meta.json"

    # -- persistence -------------------------------------------------------
    def _save_meta(self, job: Job) -> None:
        # Write atomically (temp file + rename) so a crash mid-write can never
        # leave a truncated/corrupt meta.json — readers see either the old file
        # or the complete new one.
        try:
            path = self._meta_path(job.id)
            tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
            tmp.write_text(json.dumps(job.to_dict(), indent=2))
            os.replace(tmp, path)
        except Exception:
            pass

    def _load_existing(self) -> None:
        field_names = {f.name for f in dataclasses.fields(Job)}
        for meta in sorted(self.workspace.glob("*/meta.json")):
            # One corrupt or partially-written meta.json (e.g. the server was
            # killed mid-write, or an old schema) must never block startup and
            # lose access to every other job — skip it.
            try:
                d = json.loads(meta.read_text())
                if not isinstance(d, dict):
                    continue
                d.pop("progress", None)
                job = Job(**{k: v for k, v in d.items() if k in field_names})
            except Exception:
                continue
            # A job that was mid-flight when the server stopped can't be resumed.
            if job.status in (QUEUED, RUNNING):
                job.status = FAILED
                job.error = job.error or "Interrupted by server restart."
            self.jobs[job.id] = job

    # -- submission --------------------------------------------------------
    def submit(self, payload: dict[str, Any]) -> Job:
        # Validate types up front so a malformed request is a clean 400 (and can
        # never reach the worker as e.g. a string-where-a-dict-was-expected,
        # which would crash the run thread).
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        kind = payload.get("kind")
        if kind not in ("predict", "design"):
            raise ValueError("kind must be 'predict' or 'design'")
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        # Raw server-side file writes are disabled in the demo: every input
        # arrives as inline target/spec text (validated below), and local-file
        # references are blocked, so there's no need to drop arbitrary files
        # into a job dir — and not doing so removes that whole attack surface.
        if payload.get("files"):
            raise ValueError("File uploads are disabled in the demo; paste sequences instead.")
        model = payload.get("model")
        protocol = payload.get("protocol")
        # Clamp every numeric knob into its allowed range — the client is never
        # trusted (the UI mirrors this, but this is the authority).
        params = limits.clamp_params(params, kind)
        job_id = uuid.uuid4().hex[:12]
        name = str(payload.get("name") or "").strip() or f"{kind}-{job_id[:6]}"
        job = Job(
            id=job_id, kind=kind, name=name, created_at=time.time(),
            params=params,
            model=str(model) if model is not None else None,
            protocol=str(protocol) if protocol is not None else None,
        )

        if kind == "predict":
            targets = payload.get("targets") or []
            if not isinstance(targets, list) or not targets:
                raise ValueError("predict job needs a non-empty list of targets")
            for t in targets:
                if not isinstance(t, dict) or not isinstance(t.get("content"), str):
                    raise ValueError("each target needs a string 'content' field")
            # Reject ligand-only inputs (can't fold a ligand alone) and enforce
            # the demo limits + safety rules on the actual parsed content.
            for t in targets:
                if _is_ligand_only(t["content"]):
                    raise ValueError(
                        "A ligand can't be folded on its own — include a protein, "
                        "DNA, or RNA chain for it to bind.")
            limits.check_targets(targets)
            # Only now create job dirs — a rejected submission leaves no litter.
            inputs = self._inputs_dir(job_id)
            inputs.mkdir(parents=True, exist_ok=True)
            self._out_dir(job_id).mkdir(parents=True, exist_ok=True)
            fallback = "fasta" if payload.get("input_format") == "fasta" else "yaml"
            # Sanitise names into safe, unique file stems — tt-bio keys each
            # result row and structure file by the input file's stem, so these
            # must be filesystem-safe and collision-free. The extension is chosen
            # per target from its *content*, so a mixed batch can't be misparsed.
            seen: set[str] = set()
            for i, t in enumerate(targets):
                stem = _safe_stem(Path(str(t.get("name") or "")).stem, f"target_{i + 1}")
                base, n = stem, 2
                while stem in seen:
                    stem, n = f"{base}_{n}", n + 1
                seen.add(stem)
                ext = _detect_ext(t["content"], fallback)
                (inputs / f"{stem}.{ext}").write_text(t["content"])
            job.total = len(targets)
        else:  # design
            spec = payload.get("spec")
            if not isinstance(spec, str) or not spec.strip():
                raise ValueError("design job needs a spec string")
            limits.check_design(spec)
            inputs = self._inputs_dir(job_id)
            inputs.mkdir(parents=True, exist_ok=True)
            self._out_dir(job_id).mkdir(parents=True, exist_ok=True)
            (inputs / "design.yaml").write_text(spec)

        with self.lock:
            self.jobs[job_id] = job
        self._save_meta(job)
        self.queue.put(job_id)
        return job

    # -- command construction ---------------------------------------------
    # The platform exposes a fixed, vetted set of options — it deliberately does
    # NOT forward arbitrary CLI args, device ids, or unknown params, so a request
    # can never inject low-level flags into the tt-bio subprocess.
    def _int(self, p, key):
        v = p.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    def _build_cmd(self, job: Job, controller_url: str | None = None) -> list[str]:
        p = job.params
        out = self._out_dir(job.id)
        if job.kind == "predict":
            cmd = [*TTBIO, "predict", str(self._inputs_dir(job.id)),
                   "--out_dir", str(out), "--model", job.model or "boltz2",
                   "--accelerator", "tenstorrent", "--debug", "--log",
                   "--output_format", "pdb" if p.get("output_format") == "pdb" else "cif"]
            # When a shared cluster is up, submit to it instead of starting a
            # local scheduler — the controller fans this run's targets across
            # every connected galaxy, and many such clients run concurrently.
            if controller_url:
                cmd += ["--controller", controller_url, "--run-id", job.id]
            for flag in ("use_msa_server", "fast"):
                if p.get(flag):
                    cmd.append(f"--{flag}")
            for key in ("recycling_steps", "sampling_steps", "diffusion_samples"):
                v = self._int(p, key)
                if v is not None:
                    cmd += [f"--{key}", str(v)]
            return cmd
        # design
        cmd = [*TTBIO, "gen", "run", "design.yaml", "--output", str(out),
               "--protocol", job.protocol or "protein-anything", "--debug", "--log"]
        # With a shared cluster up, design fans across the fleet exactly like
        # predict: the controller leases one shard per worker, each runs a
        # single-device gen run, and this client merges + filters the union.
        if controller_url:
            cmd += ["--controller", controller_url, "--run-id", job.id]
        for key in ("num_designs", "budget"):
            v = self._int(p, key)
            if v is not None:
                cmd += [f"--{key}", str(v)]
        if p.get("fast"):
            cmd.append("--fast")
        return cmd

    # -- worker loop -------------------------------------------------------
    def _run_loop(self) -> None:
        # One of a pool of identical threads. Concurrency (and the predict /
        # design exclusivity split) is governed by _admit/_retire, not by the
        # number of threads.
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status == CANCELED:
                continue
            url = self.cluster.submit_url() if self.cluster else None
            # Both predict and design submit to the shared controller when one is
            # up — thin clients whose compute the fleet's workers do, so they run
            # concurrently. With no cluster, a job runs locally and needs the
            # devices to itself (exclusive).
            controller_url = url
            exclusive = controller_url is None
            self._admit(exclusive)
            try:
                self._run_job(job, controller_url)
            except Exception as e:  # never let one job kill its worker thread
                job.status = FAILED
                job.error = f"Internal error: {e}"
                job.finished_at = time.time()
                self._save_meta(job)
            finally:
                self._retire(exclusive)

    def _run_job(self, job: Job, controller_url: str | None = None) -> None:
        job.status = RUNNING
        job.started_at = time.time()
        self._save_meta(job)
        cmd = self._build_cmd(job, controller_url)
        log = self._log_path(job.id)
        with open(log, "w") as logf:
            logf.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n\n")
            logf.flush()
            # Quiet third-party noise that otherwise floods the job log: the
            # huggingface_hub "Fetching N files" progress bars and tokenizer
            # fork warnings. (Spurious tt-bio warnings are fixed at the source.)
            env = {
                **os.environ,
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(self._inputs_dir(job.id)), env=env,
                    stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True,  # own process group, so cancel kills children
                )
            except Exception as e:  # pragma: no cover - launch failure
                job.status = FAILED
                job.error = f"Failed to launch tt-bio: {e}"
                job.finished_at = time.time()
                self._save_meta(job)
                return
            with self.lock:
                self.procs[job.id] = proc
            # Poll for progress until the process exits.
            while proc.poll() is None:
                self._update_progress(job)
                self._save_meta(job)
                time.sleep(1.0)
            rc = proc.returncode
        with self.lock:
            self.procs.pop(job.id, None)
        self._update_progress(job)
        job.returncode = rc
        job.finished_at = time.time()
        if job.status == CANCELED:
            pass
        elif rc == 0:
            # tt-bio exits 0 even if individual targets failed. Treat a run
            # where *every* target failed as a failed job rather than a
            # misleading "succeeded".
            if job.kind == "predict" and job.total:
                ok = self._ok_count(job)
                job.done = job.total
                if ok == 0:
                    job.status = FAILED
                    job.error = "Every target failed — see the per-target status and the log below."
                else:
                    job.status = SUCCEEDED
            else:
                job.status = SUCCEEDED
                job.done = job.total or job.done
        else:
            job.status = FAILED
            job.error = self._tail_error(job)
        self._save_meta(job)

    def _ok_count(self, job: Job) -> int:
        rd = self._results_dir(job)
        rows = (_read_json(rd / "results.json") if rd else None) or []
        return sum(1 for r in rows if isinstance(r, dict) and r.get("status") in (None, "ok"))

    # -- progress / results parsing ---------------------------------------
    def _results_dir(self, job: Job) -> Path | None:
        if job.kind == "predict":
            hits = sorted(self._out_dir(job.id).glob("boltz_results_*"))
            return hits[0] if hits else None
        return self._out_dir(job.id)

    def _update_progress(self, job: Job) -> None:
        rd = self._results_dir(job)
        if job.kind == "predict":
            if rd:
                rows = _read_json(rd / "results.json") or []
                if isinstance(rows, list):
                    job.done = len([r for r in rows if isinstance(r, dict)])
            job.stage = self._last_stage(job)
        else:
            job.stage = self._design_stage(job)

    def _last_stage(self, job: Job) -> str | None:
        text = self._tail(job, 4000).lower()
        # Best-effort: surface the most recent recognisable stage word.
        for word in ("affinity", "diffusion", "sampling", "trunk", "pairformer",
                     "msa", "featuriz", "loading", "writing"):
            idx = text.rfind(word)
            if idx != -1:
                return word
        return None

    def _design_stage(self, job: Job) -> str | None:
        text = self._tail(job, 8000).lower()
        found = None
        for stage in _DESIGN_STAGES:
            if stage.replace("_", " ") in text or stage in text:
                found = stage
        return found

    def _tail(self, job: Job, nbytes: int) -> str:
        try:
            with open(self._log_path(job.id), "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - nbytes))
                return f.read().decode("utf-8", "replace")
        except Exception:
            return ""

    # Benign teardown chatter ttnn prints at process exit — never the real error.
    _NOISE = ("nanobind", "leaked instance", "leaked type", "leaked function",
              "- leaked", "skipped remainder", "reference counting issue", "refleaks")

    def _tail_error(self, job: Job) -> str:
        # Read a generous tail and drop the ttnn nanobind teardown block, which
        # is long enough to otherwise crowd the real traceback out of the tail.
        raw = self._tail(job, 16000).splitlines()
        lines = [ln for ln in raw if ln.strip()
                 and not any(n in ln for n in self._NOISE)]
        if not lines:
            return "Job failed (see log)."
        # Prefer the most specific exception line (+ the line above for context).
        markers = ("Error:", "Exception:", "RuntimeError", "ValueError", "KeyError",
                   "TypeError", "AssertionError", "OSError", "HTTPError")
        for i in range(len(lines) - 1, -1, -1):
            if any(m in lines[i] for m in markers):
                return "\n".join(lines[max(0, i - 1): i + 1])
        return "\n".join(lines[-12:])

    # -- public read API ---------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.status == RUNNING:
            self._update_progress(job)
        d = job.to_dict()
        d["results"] = self.results(job)
        d["log_size"] = self._log_path(job_id).stat().st_size if self._log_path(job_id).exists() else 0
        return d

    def results(self, job: Job) -> dict[str, Any]:
        rd = self._results_dir(job)
        if not rd or not rd.exists():
            return {"ready": False}
        if job.kind == "predict":
            return self._predict_results(job, rd)
        return self._design_results(job, rd)

    def _predict_results(self, job: Job, rd: Path) -> dict[str, Any]:
        rows = _read_json(rd / "results.json") or []
        rows = [r for r in rows if isinstance(r, dict)]
        struct_dir = rd / "structures"
        structures: dict[str, list[str]] = {}
        if struct_dir.exists():
            for f in sorted(struct_dir.glob("*")):
                if f.suffix.lower() in (".cif", ".pdb"):
                    key = f.name.split("_model_")[0].rsplit(".", 1)[0]
                    structures.setdefault(key, []).append(f.name)
        return {"ready": bool(rows), "kind": "predict", "rows": rows, "structures": structures}

    def _design_results(self, job: Job, rd: Path) -> dict[str, Any]:
        ranked = rd / "final_ranked_designs"
        # The ranked CSV and structure dir are named for the budget
        # (final_designs_metrics_<N>.csv, final_<N>_designs), not always 30.
        metrics = sorted(ranked.glob("final_designs_metrics_*.csv"))
        csv_path = metrics[0] if metrics else (ranked / "all_designs_metrics.csv")
        if not csv_path.exists():
            return {"ready": False}
        keep = ["final_rank", "id", "designed_sequence", "num_design",
                "design_to_target_iptm", "design_ptm", "min_design_to_target_pae",
                "plip_hbonds_refolded", "delta_sasa_refolded", "quality_score",
                "file_name", "liability_num_violations"]
        designs = []
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    designs.append({k: row.get(k) for k in keep if k in row})
        except Exception:
            return {"ready": False}
        try:
            designs.sort(key=lambda r: float(r.get("final_rank") or 1e9))
        except Exception:
            pass
        # Map ranks to structure files in the budget-named dir (final_<N>_designs).
        struct_dir = next((d for d in sorted(ranked.glob("final_*_designs")) if d.is_dir()), None)
        files = sorted(struct_dir.glob("rank*.cif")) if struct_dir else []
        rank_files = {}
        for f in files:
            try:
                rank_files[int(f.name.split("_")[0].replace("rank", ""))] = f"{struct_dir.name}/{f.name}"
            except Exception:
                pass
        for d in designs:
            try:
                d["structure"] = rank_files.get(int(float(d.get("final_rank"))))
            except Exception:
                d["structure"] = None
        return {"ready": True, "kind": "design", "designs": designs}

    def structure_file(self, job_id: str, relpath: str) -> Path | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        rd = self._results_dir(job)
        if not rd:
            return None
        base = (rd / "structures") if job.kind == "predict" else (rd / "final_ranked_designs")
        target = (base / relpath).resolve()
        # Contain to the results dir. Use a real path-boundary check, not a
        # string prefix (which would let a sibling dir sharing a name-prefix,
        # e.g. <rd>_evil, slip through).
        if not target.is_relative_to(rd.resolve()):
            return None
        return target if target.exists() else None

    def log_text(self, job_id: str) -> str:
        path = self._log_path(job_id)
        return path.read_text(errors="replace") if path.exists() else ""

    def archive(self, job_id: str) -> Path | None:
        """Zip the whole results directory (structures + results.json / CSVs) for
        bulk download. Rebuilt on demand."""
        job = self.jobs.get(job_id)
        if job is None:
            return None
        rd = self._results_dir(job)
        if not rd or not rd.exists():
            return None
        zpath = self.job_dir(job_id) / "results.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in rd.rglob("*"):
                if f.is_file() and f.resolve() != zpath.resolve():
                    z.write(f, f.relative_to(rd))
        return zpath

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        # Tell the controller to cancel the run so any shards/targets already
        # leased to workers stop immediately (the run id is the job id) — killing
        # only the local orchestrator below would leave them hogging devices.
        if self.cluster is not None:
            try:
                self.cluster.cancel_run(job_id)
            except Exception:
                pass
        with self.lock:
            proc = self.procs.get(job_id)
        if job.status == QUEUED:
            job.status = CANCELED
            job.finished_at = time.time()
            self._save_meta(job)
            return True
        if proc and proc.poll() is None:
            job.status = CANCELED
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            self._save_meta(job)
            return True
        return False

    def delete(self, job_id: str) -> bool:
        self.cancel(job_id)
        with self.lock:
            self.jobs.pop(job_id, None)
        import shutil
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _safe_stem(name: str, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return s[:80] or fallback


def _is_ligand_only(content: str) -> bool:
    """A ligand only makes sense bound to a polymer; folding one alone isn't a
    structure-prediction task. (protein/DNA/RNA on their own are fine.)"""
    has_ligand = bool(re.search(r"(^|\n)\s*-?\s*ligand\s*:", content, re.I))
    has_polymer = bool(re.search(r"(^|\n)\s*-?\s*(protein|dna|rna)\s*:", content, re.I))
    return has_ligand and not has_polymer


def _detect_ext(content: str, fallback: str = "yaml") -> str:
    """Pick the right input extension from the content itself, so YAML is never
    handed to the FASTA parser (or vice-versa)."""
    s = (content or "").lstrip()
    if s.startswith(">"):
        return "fasta"
    if s.startswith("version:") or "sequences:" in s or "entities:" in s:
        return "yaml"
    return fallback
