#!/usr/bin/env python3
"""Concurrency harness shared by the GPU and the Tenstorrent legs.

One barrier, one estimator, both sides. That symmetry is the point of this pass: the
committed comparison put a 3-way-parallel TT aggregate (0.211 folds/s) against a serial
H200 number (0.163 folds/s), which is not a comparison. Aggregate throughput has to be
the same quantity measured the same way on both sides or it means nothing.

Protocol. Every worker loads its model and runs one untimed cold fold, then blocks at a
file barrier until all N workers are ready. On release each worker runs K back-to-back
timed folds and writes its per-fold (start, end) pairs. The launcher aggregates.

Timestamps are ``time.monotonic()``. On Linux that is CLOCK_MONOTONIC, one boot-relative
epoch shared by every process on the host, so comparing timestamps across worker
processes is valid (``time.perf_counter()`` is not, and wall-clock ``time.time()`` can
step under NTP).

Two estimators are reported for every point, and they have to agree:

* ``agg_folds_per_s`` = sum over workers of (its folds / its own span, first fold start to
  last fold end). The span deliberately includes whatever the worker does between folds,
  so harness overhead counts against throughput instead of vanishing. This is the headline
  number.
* ``agg_folds_per_s_window`` = sum over workers of (its duty cycle inside the common
  overlap window / its mean fold latency). Computed from the window every worker was
  provably inside, so a worker that started late, died early or idled shows up here and
  not in the first.

If the two disagree by more than 10% the point is marked not clean and must not be
published: it means the workers were not actually overlapping, so the "aggregate" is
really a partly-serial number -- exactly the bug this whole task exists to fix.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

BARRIER_TIMEOUT_S = 1800.0
_POLL_S = 0.02

# Two independent estimators of the same quantity may differ by this much before the
# point is called dirty. 10% is well above the run-to-run spread seen on either side
# (committed H200 warm folds 6.141/6.147/6.149 s) and well below any real concurrency
# effect we would want to report.
AGREEMENT_TOL = 0.10


def barrier(run_dir: Path, worker_id: int, n: int,
            timeout_s: float = BARRIER_TIMEOUT_S) -> float:
    """Announce this worker ready, block until all ``n`` are, return the release time.

    File-based rather than a socket or a multiprocessing primitive because the workers
    are independently spawned processes (that is the whole point -- separate CUDA
    contexts / separate TT device opens), and a file barrier survives them being launched
    by a shell.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"ready.{worker_id}").write_text(str(time.monotonic()))
    deadline = time.monotonic() + timeout_s
    while True:
        ready = len(list(run_dir.glob("ready.*")))
        if ready >= n:
            return time.monotonic()
        # A dead peer means this point can never be N-way concurrent, so waiting out the
        # timeout just burns clock -- on the rented GPU, literal money.
        dead = sorted(p.name for p in run_dir.glob("failed.*"))
        if dead:
            raise RuntimeError(f"peer worker failed before the barrier: {dead}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"barrier timeout: {ready}/{n} workers ready after {timeout_s}s")
        time.sleep(_POLL_S)


def mark_failed(run_dir: Path, worker_id: int, exc: BaseException) -> None:
    """Tell peers still waiting at the barrier to give up now."""
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"failed.{worker_id}").write_text(repr(exc))
    except Exception:
        pass


def write_worker_result(run_dir: Path, worker_id: int, payload: dict) -> Path:
    """Write one worker's result atomically, so the launcher never reads a half file."""
    out = run_dir / f"done.{worker_id}.json"
    tmp = run_dir / f".done.{worker_id}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, out)
    return out


def load_worker_results(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(run_dir.glob("done.*.json"),
                            key=lambda p: int(p.name.split(".")[1]))]


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-worker fold timings into one throughput figure.

    ``results`` entries need ``worker`` (an id) and ``folds`` ([[start, end], ...] in
    monotonic seconds). Anything else is ignored.
    """
    per, pooled = [], []
    for r in results:
        folds = [(float(s), float(e)) for s, e in r["folds"]]
        if not folds:
            continue
        durs = [e - s for s, e in folds]
        busy = sum(durs)
        span = folds[-1][1] - folds[0][0]
        pooled += durs
        per.append(dict(
            worker=r["worker"], n_folds=len(folds), busy_s=round(busy, 3),
            span_s=round(span, 3),
            # Below 1.0 means the worker spent part of its loop not folding: real lost
            # throughput, already charged to it by rate = folds / span.
            duty_cycle=round(busy / span, 4) if span > 0 else 0.0,
            mean_latency_s=busy / len(folds),
            median_latency_s=round(statistics.median(durs), 4),
            rate_folds_per_s=len(folds) / span if span > 0 else 0.0,
            first_start=folds[0][0], last_end=folds[-1][1], folds=folds,
        ))
    if not per:
        return dict(clean=False, reason="no worker produced a timed fold")

    agg_rate_sum = sum(p["rate_folds_per_s"] for p in per)
    w_start = max(p["first_start"] for p in per)
    w_end = min(p["last_end"] for p in per)
    window_s = w_end - w_start

    agg_window = None
    coverage = []
    if window_s > 0:
        for p in per:
            inside = sum(max(0.0, min(e, w_end) - max(s, w_start)) for s, e in p["folds"])
            coverage.append(inside / window_s)
        agg_window = sum(c / p["mean_latency_s"] for c, p in zip(coverage, per))

    # How far apart the workers finished. A skew above one fold means the last stretch was
    # not N-way concurrent at all, so a naive aggregate would be inflated.
    finishes = [p["last_end"] for p in per]
    tail_skew_s = max(finishes) - min(finishes)
    mean_lat = statistics.mean(pooled)

    reasons = []
    if agg_window is None:
        reasons.append("no overlap window: the workers never ran at the same time")
    else:
        rel = abs(agg_window - agg_rate_sum) / agg_rate_sum
        if rel > AGREEMENT_TOL:
            reasons.append(f"estimators disagree by {rel:.1%} (> {AGREEMENT_TOL:.0%})")
    if tail_skew_s > mean_lat:
        reasons.append(f"tail skew {tail_skew_s:.1f}s exceeds one fold ({mean_lat:.1f}s)")
    if min(p["n_folds"] for p in per) < 3:
        reasons.append("a worker ran fewer than 3 timed folds")

    ps = sorted(pooled)
    return dict(
        n_workers=len(per),
        agg_folds_per_s=round(agg_rate_sum, 5),
        agg_folds_per_s_window=round(agg_window, 5) if agg_window else None,
        window_s=round(window_s, 3),
        # Monotonic bounds of the all-workers-busy window, so a device-telemetry
        # sampler taken on the same clock can be restricted to it.
        window_start=w_start, window_end=w_end,
        min_worker_coverage=round(min(coverage), 4) if coverage else None,
        min_duty_cycle=round(min(p["duty_cycle"] for p in per), 4),
        tail_skew_s=round(tail_skew_s, 3),
        latency_min_s=round(ps[0], 3),
        latency_median_s=round(statistics.median(ps), 3),
        latency_max_s=round(ps[-1], 3),
        total_folds=len(pooled),
        clean=not reasons, reason="; ".join(reasons) or None,
        per_worker=[{k: (round(v, 5) if isinstance(v, float) else v)
                     for k, v in p.items() if k != "folds"} for p in per],
    )


# --------------------------------------------------------------------------------------
# self-test: the aggregation math and the real barrier, no hardware, ~2 s.
# --------------------------------------------------------------------------------------

def _fake(worker, n_folds, start, latency, gap=0.0):
    folds, t = [], start
    for _ in range(n_folds):
        folds.append([t, t + latency])
        t += latency + gap
    return dict(worker=worker, folds=folds)


def _self_test() -> int:
    import multiprocessing as mp
    import tempfile

    # 1. Four workers, 5 folds of exactly 2 s each, all in lockstep -> 2.0 folds/s.
    a = aggregate([_fake(i, 5, 100.0, 2.0) for i in range(4)])
    assert a["clean"], a
    assert abs(a["agg_folds_per_s"] - 2.0) < 1e-4, a
    assert abs(a["agg_folds_per_s_window"] - 2.0) < 1e-4, a
    assert a["latency_median_s"] == 2.0

    # 2. Same but staggered starts within one fold: still 2.0, still clean. This is the
    #    realistic case -- the barrier releases workers a few ms apart.
    a = aggregate([_fake(i, 6, 100.0 + 0.05 * i, 2.0) for i in range(4)])
    assert a["clean"], a
    assert abs(a["agg_folds_per_s"] - 2.0) < 1e-4, a
    assert abs(a["agg_folds_per_s_window"] - 2.0) < 0.05, a

    # 3. Unequal speeds: 1 s and 3 s workers -> 1/1 + 1/3 = 1.3333 folds/s.
    a = aggregate([_fake(0, 9, 100.0, 1.0), _fake(1, 3, 100.0, 3.0)])
    assert abs(a["agg_folds_per_s"] - (1 + 1 / 3)) < 1e-4, a
    assert a["clean"], a

    # 4. A straggler that never overlaps must be caught, not averaged away -- this is the
    #    failure mode that produced the number being corrected.
    a = aggregate([_fake(0, 5, 100.0, 2.0), _fake(1, 5, 130.0, 2.0)])
    assert not a["clean"], a

    # 5. A worker idling half its loop: both estimators must charge it for the idle time
    #    (5/18 + 5/10 = 0.7778 folds/s), and its duty cycle must expose the overhead.
    a = aggregate([_fake(0, 5, 100.0, 2.0), _fake(1, 5, 100.0, 2.0, gap=2.0)])
    assert abs(a["agg_folds_per_s"] - (5 / 18 + 0.5)) < 1e-4, a
    assert abs(a["agg_folds_per_s_window"] - a["agg_folds_per_s"]) / a["agg_folds_per_s"] < 0.1, a
    assert a["min_duty_cycle"] < 0.9, a

    # 6. Too few folds is not a measurement.
    a = aggregate([_fake(i, 2, 100.0, 2.0) for i in range(2)])
    assert not a["clean"], a

    # 7. The real barrier across real processes.
    def _w(d, i, n, q):
        try:
            t = barrier(Path(d), i, n, timeout_s=30)
            time.sleep(0.2)
            write_worker_result(Path(d), i, dict(worker=i, released=t,
                                                 folds=[[t, t + 0.2]]))
            q.put(("ok", i, t))
        except Exception as exc:                       # pragma: no cover - failure path
            q.put(("err", i, repr(exc)))

    with tempfile.TemporaryDirectory() as d:
        q = mp.Queue()
        procs = [mp.Process(target=_w, args=(d, i, 3, q)) for i in range(3)]
        for p in procs:
            p.start()
            time.sleep(0.15)          # stagger the launches; the barrier must absorb it
        for p in procs:
            p.join(timeout=40)
        out = [q.get(timeout=5) for _ in range(3)]
        assert all(o[0] == "ok" for o in out), out
        rel = [o[2] for o in out]
        assert max(rel) - min(rel) < 0.5, f"barrier released {max(rel)-min(rel):.3f}s apart"
        assert len(load_worker_results(Path(d))) == 3

    print("conc.py self-test PASS")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_self_test() if "--self-test" in sys.argv else
                     print(__doc__) or 0)
