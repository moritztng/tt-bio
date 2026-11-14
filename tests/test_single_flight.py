"""A burst of jobs that share a sequence searches it once, not once per job.

On 2026-11-13 a 29-job OpenDDE burst on one Galaxy needed 7 distinct chain searches and ran each
of them in every job that wanted it, all at the same moment. ``cache.single_flight`` gives each
search key one flock: a free key is searched at once, a held key is waited for and read from the
cache. These cases run real processes against one lock directory, no device and no database.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

from tt_bio.cache import cached, publish_text, single_flight


def _job(lock_dir: str, keys: list[str], log: str, fail: bool, go) -> None:
    out = Path(lock_dir)
    go.wait()

    def search(todo):
        with open(log, "a") as f:
            f.write(" ".join(todo) + "\n")
        time.sleep(0.4)
        if fail:
            raise RuntimeError("search failed")
        for k in todo:
            publish_text(out / f"{k}.a3m", f">{k}\n")

    try:
        single_flight(out, keys, lambda k: not cached(out / f"{k}.a3m"), search)
    except RuntimeError:
        pass


def _run(tmp_path: Path, jobs: list[tuple[list[str], bool]]) -> list[list[str]]:
    ctx = mp.get_context("fork")
    go = ctx.Event()
    log = tmp_path / "searches.log"
    procs = [ctx.Process(target=_job, args=(str(tmp_path), keys, str(log), fail, go)) for keys, fail in jobs]
    for p in procs:
        p.start()
    go.set()
    for p in procs:
        p.join(30)
        assert p.exitcode == 0
    return [line.split() for line in log.read_text().splitlines()]


def test_shared_keys_are_searched_once_across_a_burst(tmp_path: Path):
    # one antigen "ag" in every job, binders partly repeated: 4 distinct keys over 12 jobs
    jobs = [(["ag", f"b{i % 3}"], False) for i in range(12)]
    calls = _run(tmp_path, jobs)
    searched = sorted(k for c in calls for k in c)
    assert searched == ["ag", "b0", "b1", "b2"]
    assert all(cached(tmp_path / f"{k}.a3m") for k in searched)


def test_distinct_keys_still_run_side_by_side(tmp_path: Path):
    t0 = time.time()
    calls = _run(tmp_path, [([f"k{i}"], False) for i in range(6)])
    assert sorted(k for c in calls for k in c) == [f"k{i}" for i in range(6)]
    assert time.time() - t0 < 6 * 0.4   # not serialised


def test_a_failed_search_leaves_the_key_to_its_waiter(tmp_path: Path):
    # the first job takes the lock and fails; the second waits, finds nothing, searches itself
    ctx = mp.get_context("fork")
    go = ctx.Event()
    log = tmp_path / "searches.log"
    first = ctx.Process(target=_job, args=(str(tmp_path), ["ag"], str(log), True, go))
    first.start()
    go.set()
    time.sleep(0.1)
    second = ctx.Process(target=_job, args=(str(tmp_path), ["ag"], str(log), False, go))
    second.start()
    for p in (first, second):
        p.join(30)
    assert log.read_text().split() == ["ag", "ag"]
    assert cached(tmp_path / "ag.a3m")


def test_a_cached_key_is_not_searched(tmp_path: Path):
    publish_text(tmp_path / "ag.a3m", ">ag\n")
    called = []
    single_flight(tmp_path, ["ag"], lambda k: not cached(tmp_path / f"{k}.a3m"), called.append)
    assert called == []
    assert not any(n.endswith(".lock") and os.path.getsize(tmp_path / n) for n in os.listdir(tmp_path))
