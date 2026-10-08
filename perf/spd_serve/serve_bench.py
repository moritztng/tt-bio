"""Folds per chip-hour through JapanFold's own serving loop on one Galaxy chip.

    python serve_bench.py --out RUN --chip N --inputs c730,l512 --copies 3 [--model protenix-v2]
                          [--engine DIR] [--env K=V,...] [--fast] [--port 18765]

What runs is what a JapanFold agent runs, minus the API: a private loopback ``tt-bio controller``,
the agent's own ``python -m japanfold.chipworker serve`` on chip N (scrub, run marker, warm-up),
and one ``tt-bio predict --controller`` run holding every target, so the worker always has the
next target queued. ``--engine`` puts a tt-bio checkout first on PYTHONPATH for the controller,
the worker and the client; the log records which tt_bio each one imported.

The first target of each distinct input is the cold fold of that size on this worker and is
reported apart. Throughput is measured over the warm ones: from the first warm target's start
to the last one's end, so every gap between folds (complete, lease, scrub, featurise, write)
is inside it. A fold starts when the worker leases it and ends when the controller has its
result; ``fold`` is the engine's own runtime_s, and the rest of the served time is host work on the
chip's clock. AICLK of the chip's node and the host's loadavg are sampled every 0.5 s.

The run carries one account, the way an agent task does, so the chipworker scrubs the chip once
for it rather than before every fold (it scrubs whenever a run has no account). Each scrub's cost
is in ``ws/chips/<chip>.json``.

``--reduce`` recomputes the summary of an earlier ``--out`` from its serve.jsonl and results.json.

Run it on a chip CHIPS.md gives you, with the box's env sourced (``. ~/japanfold/env.sh``), and
``TT_BIO_LEASE_HOLDER`` set the way CHIPS.md says. Inputs and their MSA cache come from
``~/spd-data`` (see state/spd/BOARD.md, "How to post").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--reduce", action="store_true", help="only recompute the summary of an existing --out")
ap.add_argument("--chip")
ap.add_argument("--inputs")
ap.add_argument("--copies", type=int, default=3)
ap.add_argument("--model", default="protenix-v2")
ap.add_argument("--engine", type=Path)
ap.add_argument("--env", default="")
ap.add_argument("--fast", action="store_true")
ap.add_argument("--samples", type=int)
ap.add_argument("--port", type=int, default=18765)
ap.add_argument("--share", type=int, default=32)
ap.add_argument("--data", type=Path, default=Path("~/spd-data").expanduser())
ap.add_argument("--app", type=Path, default=Path("~/japanfold/app").expanduser())
a = ap.parse_args()

out = a.out.resolve()
out.mkdir(parents=True, exist_ok=True)
log = open(out / ("reduce.jsonl" if a.reduce else "serve.jsonl"), "a")


def emit(**kw):
    kw["t"] = round(time.time(), 3)
    log.write(json.dumps(kw) + "\n")
    log.flush()
    print(json.dumps(kw), flush=True)


env = dict(os.environ)
env.update(dict(kv.split("=", 1) for kv in a.env.split(",") if kv))
env["PYTHONPATH"] = os.pathsep.join(p for p in (str(a.engine.resolve()) if a.engine else "", str(a.app),
                                                env.get("PYTHONPATH", "")) if p)
py = sys.executable
url = f"http://127.0.0.1:{a.port}"
procs: list[subprocess.Popen] = []


def spawn(name, cmd, **kw):
    p = subprocess.Popen(cmd, env=env, stdout=open(out / f"{name}.log", "a"), stderr=subprocess.STDOUT,
                         start_new_session=True, **kw)
    procs.append(p)
    emit(ev="spawn", name=name, pid=p.pid, cmd=cmd)
    return p


def stop_all():
    # SIGINT first: the worker closes its chip on it. SIGTERM only if it is still there a minute later.
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 20)):
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
        t_end = time.time() + wait
        while time.time() < t_end and any(p.poll() is None for p in procs):
            time.sleep(0.5)


def get(path):
    with urllib.request.urlopen(url + path, timeout=10) as r:
        return json.loads(r.read())


if not a.reduce:
    which = subprocess.run([py, "-c", "import tt_bio, subprocess, os; d=os.path.dirname(os.path.dirname(tt_bio.__file__));"
                            "print(d, subprocess.run(['git','-C',d,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip())"],
                           env=env, capture_output=True, text=True).stdout.split()
    emit(ev="engine", path=which[0] if which else None, sha=which[1] if len(which) > 1 else None, env=a.env,
         fast=a.fast, model=a.model, chip=a.chip, loadavg=open("/proc/loadavg").read().split()[:3])

    # Clock of the node the worker opens. Which node that is shows up in the worker's open fds.
    clock: list[tuple[float, int]] = []
    load: list[float] = []
    node: list[int] = []


    def sampler(worker_pid):
        while True:
            # UMD discovery opens every chip of the Galaxy for a moment, so the first fd seen can be any node.
            # The chip that folds is the one node the worker holds alone; follow it if it changes.
            held = set()
            for fd in glob.glob(f"/proc/{worker_pid}/fd/*") + [f for c in _children(worker_pid)
                                                             for f in glob.glob(f"/proc/{c}/fd/*")]:
                try:
                    tgt = os.readlink(fd)
                except OSError:
                    continue
                if tgt.startswith("/dev/tenstorrent/"):
                    held.add(int(tgt.rsplit("/", 1)[1]))
            if len(held) == 1 and held != set(node):
                node[:] = list(held)
                emit(ev="node", node=node[0])
            if node:
                try:
                    v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{node[0]}/tt_aiclk").read_text().split()[0])
                    if 100 <= v <= 3000:    # a dead ARC answers 0xFFFFFFFF without raising
                        clock.append((time.time(), v))
                except (OSError, ValueError):
                    pass
            load.append(float(open("/proc/loadavg").read().split()[0]))
            time.sleep(0.5)


    def _children(pid):
        try:
            return [int(c) for c in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
        except OSError:
            return []


    try:
        spawn("controller", [py, "-m", "tt_bio.main", "controller", "--port", str(a.port), "--no-local-workers",
                             "--accelerator", "tenstorrent"], cwd=out)
        for _ in range(120):
            try:
                get("/cluster")
                break
            except Exception:
                time.sleep(0.5)
        ws = out / "ws"
        worker = spawn("worker", [py, "-m", "japanfold.chipworker", "serve", "--connect", url, "--chip", str(a.chip),
                                  "--share", str(a.share), "--workspace", str(ws)], cwd=a.app)
        threading.Thread(target=sampler, args=(worker.pid,), daemon=True).start()
        t_up = time.time()
        while True:
            if worker.poll() is not None:
                raise SystemExit(f"worker exited rc={worker.returncode}; see {out}/worker.log")
            c = get("/cluster")
            if c.get("online_workers") or c.get("workers"):
                break
            time.sleep(1)
        emit(ev="worker_online", after_s=round(time.time() - t_up, 1))

        # One run, every target in it: copy k of input X is X_k.yaml, its MSAs in the shared cache.
        names = a.inputs.split(",")
        tdir = out / "targets"
        shutil.rmtree(tdir, ignore_errors=True)
        tdir.mkdir()
        order = [f"{n}_{k}" for k in range(a.copies) for n in names]
        for t in order:
            shutil.copy(a.data / "inputs" / f"{t.rsplit('_', 1)[0]}.yaml", tdir / f"{t}.yaml")
        cmd = [py, "-m", "tt_bio.main", "predict", str(tdir), "--controller", url, "--model", a.model,
               "--accelerator", "tenstorrent", "--output_format", "cif", "--msa_dir", str(a.data / "msa"),
               "--msa_db_path", str(Path("~/japanfold/msa/db").expanduser()), "--out_dir", str(out / "pred")]
        if a.fast:
            cmd.append("--fast")
        if a.samples:
            cmd += ["--diffusion_samples", str(a.samples)]
        run_id = f"spd-serve-{int(time.time())}"
        cmd += ["--run-id", run_id]
        (ws / "runs").mkdir(parents=True, exist_ok=True)
        (ws / "runs" / run_id).write_text("spd-serve\n")
        t_sub = time.time()
        client = spawn("client", cmd, cwd=out)
        # Every job's status and live stage, every 0.5 s: the first time a (job, status, stage) is seen is
        # when it happened, to within the poll. That times each fold's phases as served.
        seen: dict = {}
        while True:
            try:
                for j in get(f"/runs/{run_id}/jobs").get("jobs", []):
                    key = (j["id"], j["status"], j.get("stage"))
                    if key not in seen:
                        seen[key] = time.time()
                        emit(ev="job", id=j["id"], status=j["status"], stage=j.get("stage"))
            except Exception:
                pass
            if client.poll() is not None:
                break
            time.sleep(0.5)
        emit(ev="client_done", rc=client.returncode, wall_s=round(time.time() - t_sub, 1))
    finally:
        stop_all()
    emit(ev="samples", node=node[0] if node else None,
         aiclk_median=statistics.median(v for _, v in clock) if clock else None,
         aiclk_min=min((v for _, v in clock), default=None),
         load_median=statistics.median(load) if load else None, load_max=max(load, default=None))


def reduce() -> None:
    """Per fold, from the run's own records: lease to result as the controller saw it (job events
    in serve.jsonl), and the engine's runtime_s and load_s (results.json)."""
    ev = [json.loads(l) for l in open(out / "serve.jsonl") if l.strip()]
    ev = ev[next((k for k in range(len(ev) - 1, -1, -1) if ev[k]["ev"] == "engine"), 0):]  # the last run
    first: dict = {}
    for e in ev:
        if e["ev"] == "job":
            first.setdefault((e["id"], e["status"], e.get("stage")), e["t"])
    rows = {r["id"]: r for p in out.glob("pred/*/results.json") for r in json.load(open(p))}
    jobs: dict = {}
    for (jid, status, stage), t in first.items():
        d = jobs.setdefault(jid, {"stages": {}})
        if status in ("ok", "failed"):
            d["end"], d["status"] = min(t, d.get("end", t)), status
        elif status == "running":           # leased; "pending" is only queued behind other targets
            d["start"] = min(t, d.get("start", t))
            if stage:
                d["stages"][stage] = min(t, d["stages"].get(stage, t))
    names_seen, warm = set(), []
    for jid, d in sorted(jobs.items(), key=lambda kv: kv[1].get("start", float("inf"))):
        base = jid.rsplit("_", 1)[0]
        d["cold"] = base not in names_seen
        names_seen.add(base)
        r = rows.get(jid, {})
        d["runtime_s"], d["load_s"] = r.get("runtime_s"), r.get("load_s")
        if "start" in d and "end" in d:
            d["served_s"] = d["end"] - d["start"]
            if not d["cold"] and d.get("status") == "ok":
                warm.append(d)
        st = sorted(d["stages"].items(), key=lambda kv: kv[1])
        marks = st + [("end", d.get("end"))]
        d["phase_s"] = {s: round(marks[k + 1][1] - t, 1) for k, (s, t) in enumerate(st) if marks[k + 1][1]}
    # Runs before 2026-10-08 22Z have no samples line; their old summary carries the clock.
    smp = next((e for e in ev if e["ev"] == "samples"), None) or next(
        ({"aiclk_median": e["aiclk_median"], "aiclk_min": e["aiclk_min"], "node": e["node"]}
         for e in ev if e["ev"] == "summary"), {})
    summary = {"ev": "summary", "targets": len(jobs), "ok": sum(d.get("status") == "ok" for d in jobs.values()),
               "aiclk_median": smp.get("aiclk_median"), "aiclk_min": smp.get("aiclk_min"),
               "loadavg_median": smp.get("load_median"), "loadavg_max": smp.get("load_max"),
               "node": smp.get("node"), "nodes_seen": sorted({e["node"] for e in ev if e["ev"] == "node"})}
    if warm:
        # A chip-hour of serving is spent on lease-to-result plus the gap to the next lease.
        # Each size is its own line: the mix of sizes is the customer's, not ours.
        for jid, d in jobs.items():
            d["base"] = jid.rsplit("_", 1)[0]
        nxt = {}
        order = sorted((d for d in jobs.values() if "start" in d), key=lambda d: d["start"])
        for x, y in zip(order, order[1:]):
            nxt[id(x)] = max(0.0, y["start"] - x["end"]) if "end" in x else 0.0
        per = {}
        for d in warm:
            unit = d["served_s"] + nxt.get(id(d), 0.0)
            per.setdefault(d["base"], []).append((unit, d["runtime_s"] or 0.0))
        summary["sizes"] = {b: {"n": len(v), "served_s": round(statistics.mean(u for u, _ in v), 1),
                                "engine_runtime_s": round(statistics.mean(r for _, r in v), 1),
                                "host_s_on_chip_clock": round(statistics.mean(u - r for u, r in v), 1),
                                "folds_per_chip_hour": round(3600 / statistics.mean(u for u, _ in v), 2),
                                "usd_per_fold_at_0_26": round(0.26 * statistics.mean(u for u, _ in v) / 3600, 4)}
                            for b, v in sorted(per.items())}
    emit(**summary)
    emit(ev="targets", rows=[{"id": jid, "cold": d["cold"], "status": d.get("status"),
                              "served_s": round(d["served_s"], 1) if "served_s" in d else None,
                              "runtime_s": d["runtime_s"], "load_s": d["load_s"], "phase_s": d["phase_s"]}
                             for jid, d in sorted(jobs.items(), key=lambda kv: kv[1].get("start", 0))])


reduce()
