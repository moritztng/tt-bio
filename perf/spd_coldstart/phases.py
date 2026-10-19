"""Split one traced `tt-bio predict` call (trace-*.jsonl from trace/sitecustomize.py) into phases.

    python3 perf/spd_coldstart/phases.py cs2/boltz2/trace-cold-2.jsonl cs2/boltz2/trace-ctl-2.jsonl

A local run (parent spawns a worker) prints parent import, pre-dispatch, worker spawn + import, the worker's startup
device open, close + lease, load_model, the job, the parent noticing the result, stopping the worker and the tail.
A `--controller` run prints parent import, pre-dispatch, the streamed job and the tail. Seconds, wall clock.
"""
import json
import sys


def phases(path):
    ev = [json.loads(line) for line in open(path)]
    t0 = min(e["t"] for e in ev)
    end = max(e["t"] for e in ev) - t0

    def at(fn, kind, pid=None, last=False):
        hits = [e["t"] - t0 for e in ev if e.get("fn") == fn and e["ev"] == kind and pid in (None, e["pid"])]
        return hits[-1] if last else hits[0]

    starts = [e for e in ev if e["ev"] == "proc_start"]
    parent = starts[0]["pid"]
    imported = next(e["t"] - t0 for e in ev if e["ev"] == "imported" and e["mod"] == "tt_bio.main" and e["pid"] == parent)
    out = {"total": end, "parent_import": imported}
    workers = [e for e in starts if "--multiprocessing-fork" in e.get("argv", [])]
    if not workers:
        s0, s1 = at("tt_bio.main._stream_run", "enter"), at("tt_bio.main._stream_run", "leave")
        out.update(pre_dispatch=s0 - imported, stream=s1 - s0, tail=end - s1)
        return out
    w, spawned = workers[0]["pid"], workers[0]["t"] - t0
    loop = at("tt_bio.worker.run_worker_loop", "enter", w)
    depth, opened = 0, None
    for e in ev:  # the outermost get_device span that starts the worker loop
        if e["pid"] == w and e.get("fn") == "tt_bio.tenstorrent.get_device":
            depth += 1 if e["ev"] == "enter" else -1
            if depth == 0:
                opened = e["t"] - t0
                break
    l0 = at("tt_bio.worker._WorkerState.load_model", "enter", w)
    l1 = at("tt_bio.worker._WorkerState.load_model", "leave", w)
    job = at("tt_bio.worker._execute_job", "leave", w, last=True)
    seen = at("tt_bio.main._write_job_outputs", "enter", parent)
    s0 = at("tt_bio.main._stop_worker_processes", "enter", parent)
    s1 = at("tt_bio.main._stop_worker_processes", "leave", parent)
    out.update(pre_dispatch=spawned - imported, worker_spawn_import=loop - spawned, startup_open=opened - loop,
               close_lease=l0 - opened, load_model=l1 - l0, job=job - l1, notice=seen - job, stop_workers=s1 - s0,
               tail=end - s1)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, " ".join(f"{k}={v:.2f}" for k, v in phases(p).items()))
