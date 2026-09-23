#!/usr/bin/env python3
"""Fan design jobs across chips, one job per chip, and report the aggregate rate.

This is the row's data-parallelism instrument. Throughput on a Galaxy is not one chip's
seconds times 27 -- the 27 chips share one 64-core host, and every designer does real host
work (featurisation, the mmCIF writer, boltzgen's filtering). So the question "designs per
hour across the usable chips" has to be MEASURED with several chips running at once, and the
per-chip rate compared against the same job run alone.

    python3 perf/mgxscale/fan.py --plan perf/mgxscale/plans/px_batch.txt \
        --out perf/mgxscale/results/px_batch.jsonl

A plan file is one job per line: the arguments to `job.py`, minus --card/--out/--holder.
Blank lines and #-comments are skipped. A job whose chip cannot be found waits rather than
failing: losing a race for a chip is contention, not a result (`perf/mgxdesign/walk.py`).
"""
import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from perf.mgxscale.job import free_cards  # noqa: E402

PY = os.environ.get("LADDER_PY") or sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holder", default="worker:mgx-design-scale")
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--work", default=str(pathlib.Path.home() / "mgxscale-work"))
    ap.add_argument("--wait-s", type=int, default=120, help="poll period when no chip is free")
    ap.add_argument("--retries", type=int, default=6, help="requeues allowed per job on contention")
    ap.add_argument("--barrier", type=int, default=0, metavar="N",
                    help="wait until N chips are free, then launch N jobs together. A "
                         "data-parallelism arm measured by starting four jobs as chips "
                         "happen to free is four staggered solo runs, not a fan.")
    ap.add_argument("--lines", default="", metavar="N[,N...]",
                    help="run only these 1-based plan entries (comments not counted). A fan "
                         "parked on whglx cannot see the pc-side quiet-window gate, so a pass "
                         "that must not straddle the window launches one entry and lets the "
                         "fan exit, rather than leaving it queued on the rest.")
    args = ap.parse_args()

    jobs = [l.strip() for l in pathlib.Path(args.plan).read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")]
    if args.lines:
        want = [int(x) for x in args.lines.split(",") if x.strip()]
        bad = [n for n in want if not 1 <= n <= len(jobs)]
        if bad:
            sys.exit(f"--lines {bad} outside 1..{len(jobs)} for {args.plan}")
        jobs = [jobs[n - 1] for n in want]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[fan] {len(jobs)} job(s), cap {args.max_concurrent}, out {out}", flush=True)

    live: list[tuple[subprocess.Popen, int, str]] = []
    queue = list(enumerate(jobs))
    mine: set[int] = set()
    retries: dict[str, int] = {}
    if args.barrier:
        # Hold until the whole fan can start at once. Without this the first job runs alone
        # for most of its life and the per-chip rate it reports is a solo rate wearing a
        # fan's label.
        waited = 0
        while len(free_cards()) < args.barrier:
            print(f"[fan] barrier: {len(free_cards())}/{args.barrier} chips free, waited "
                  f"{waited}s", flush=True)
            time.sleep(args.wait_s)
            waited += args.wait_s
        print(f"[fan] barrier met after {waited}s", flush=True)

    while queue or live:
        while queue and len(live) < args.max_concurrent:
            free = [c for c in free_cards() if c not in mine]
            if not free:
                break
            idx, spec = queue.pop(0)
            card = free[0]
            mine.add(card)
            cmd = [PY, "-u", str(ROOT / "perf/mgxscale/job.py")] + shlex.split(spec) + [
                "--card", str(card), "--out", str(out), "--holder", args.holder,
                "--work", args.work]
            log = pathlib.Path(args.work) / f"fan_{idx}_card{card}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = log.open("w")
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            live.append((p, card, spec))
            print(f"[fan] +card{card} pid{p.pid}: {spec}", flush=True)
        if not live:
            time.sleep(args.wait_s)
            continue
        time.sleep(15)
        for ent in list(live):
            p, card, spec = ent
            if p.poll() is not None:
                live.remove(ent)
                mine.discard(card)
                # 75 is the engine's own refusal to open a chip another row took between the
                # flock probe and the open. Nothing ran, so nothing was recorded and the job
                # goes back in the queue -- counting it would put a fleet collision in a rate.
                if p.returncode == 75 and retries.get(spec, 0) < args.retries:
                    retries[spec] = retries.get(spec, 0) + 1
                    queue.append((-1, spec))
                    print(f"[fan] -card{card} CONTENTION, requeued "
                          f"({retries[spec]}/{args.retries}): {spec}", flush=True)
                    continue
                print(f"[fan] -card{card} rc={p.returncode}: {spec}", flush=True)

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    n = sum(r.get("n_designs", 0) for r in rows)
    span = time.time() - t0
    print(f"[fan] done: {n} design(s) from {len(rows)} job(s) in {span / 3600:.2f} h "
          f"= {3600 * n / span:.2f} designs/h aggregate", flush=True)


if __name__ == "__main__":
    main()
