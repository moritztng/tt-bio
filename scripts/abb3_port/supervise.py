#!/usr/bin/env python3
"""The supervisor: keeps the reproduction alive across watchdog resets, without a human.

    PYTHONPATH=$PWD python3 scripts/abb3_port/supervise.py \
        --out runs/base --steps 193512 --chips 0,1

**Why this exists rather than a note in the runbook.** qb2 took 5 watchdog resets in one day
(2026-09-13, gaps down to 48 minutes) while running four cards at full tilt, which is this run's
load profile. A 9-to-13-day run therefore contains roughly 9-47 process deaths, and each one
kills the ranks outright with no chance to save anything. So the run is a pair: worker processes
that checkpoint on a wall-clock cadence, and this, which notices they are gone and starts them
again from the last checkpoint.

**All ranks restart together, always.** A data-parallel step blocks until every rank has written
its gradient, so one dead rank stalls the survivors until their rendezvous times out. Restarting
just the dead one would leave it resuming from a checkpoint while the survivors carry state from
after it -- the "restored three ranks and re-initialised the fourth" failure, which shows a
healthy loss curve for a model nobody asked for. Killing the whole world and resuming all of it
from one file is the only arrangement where the per-rank master hashes can agree afterwards, and
they are checked every step.

**A host reboot outlives this process too, so the reboot hook is part of the run and is
explicit.** ``--print-reboot-hook`` emits the crontab line that restarts the supervisor after a
reset reboots the box. It is printed rather than installed: a crontab entry that outlives its
campaign is a recorded failure on this fleet, so installing it is a decision taken when the run
starts and undone when it ends.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_ID_ENV = "ABB3_RUN_ID"


def launch(rank: int, chips: list, args) -> subprocess.Popen:
    env = dict(os.environ)
    chip = chips[rank]
    # The grant and the pin, on the command that opens the device. An unpinned open brings up
    # every visible chip, so this is not belt-and-braces -- without it one rank takes the box.
    env["TT_VISIBLE_DEVICES"] = str(chip)
    env["TT_BIO_LEASE_CARDS"] = ",".join(str(c) for c in chips)
    env.setdefault("TT_BIO_LEASE_HOLDER", "worker:train-b3-train")
    env["PYTHONPATH"] = str(REPO)
    # So each rank's cotenancy sampler can tell a SIBLING rank from a foreign process. Via the
    # environment and not the command line, because `device_holders` truncates a cmdline at 90
    # characters and a rank's script name falls off the end of a worktree path -- which made
    # every 2-chip measurement report itself as contended by its own sibling.
    env[RUN_ID_ENV] = args.run_id
    cmd = [sys.executable, str(HERE / "repro.py"),
           "--out", args.out, "--steps", str(args.steps),
           "--global-batch", str(args.global_batch), "--micro", str(args.micro),
           "--tokens", str(args.tokens), "--blocks", str(args.blocks),
           "--seed", str(args.seed), "--rank", str(rank), "--world", str(len(chips)),
           "--chips", ",".join(str(c) for c in chips),
           "--rendezvous", args.rendezvous,
           "--checkpoint-minutes", str(args.checkpoint_minutes),
           "--data", args.data]
    if args.max_seconds:
        cmd += ["--max-seconds", str(args.max_seconds)]
    if args.kill_at and rank == args.kill_rank:
        cmd += ["--kill-at", str(args.kill_at)]
    logs = Path(args.out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fh = open(logs / f"rank{rank}.log", "a", buffering=1)
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(REPO), start_new_session=True)
    proc._log = fh  # keep the handle alive for the process's lifetime
    return proc


def kill_all(procs: list, why: str) -> None:
    """SIGKILL the process GROUP of every live rank, by explicit pid.

    The group and not the pid alone, because a rank that has opened a device leaves helper
    threads and a ttnn dispatch process behind; killing the outer pid alone leaves an orphan
    still holding the card, which is a recorded failure on this fleet. Never a pattern match --
    a pkill on a name matches other workers' processes on a shared box.
    """
    for p in procs:
        if p.poll() is None:
            print(f"[sup] killing pid {p.pid} ({why})", flush=True)
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError) as e:
                print(f"[sup] pid {p.pid}: {e}", flush=True)
    for p in procs:
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print(f"[sup] pid {p.pid} did not reap in 60s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=193_512)
    ap.add_argument("--chips", default="0")
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rendezvous", default="/dev/shm/abb3-dp")
    ap.add_argument("--checkpoint-minutes", type=float, default=30.0)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--max-restarts", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--settle", type=float, default=20.0,
                    help="seconds between a death and the restart, for the card to come back")
    ap.add_argument("--kill-at", type=int, default=0, help="demo: kill one rank at this step")
    ap.add_argument("--kill-rank", type=int, default=0)
    ap.add_argument("--print-reboot-hook", action="store_true")
    args = ap.parse_args()
    # Derived from the output directory rather than random, so a supervisor restarted after a
    # reboot stamps the SAME id and still recognises ranks it did not itself launch.
    args.run_id = hashlib.blake2b(str(Path(args.out).resolve()).encode(),
                                  digest_size=8).hexdigest()

    chips = [int(c) for c in args.chips.split(",")]
    out = Path(args.out)
    if args.print_reboot_hook:
        print(f"@reboot cd {REPO} && PYTHONPATH={REPO} {sys.executable} "
              f"{HERE / 'supervise.py'} --out {out} --steps {args.steps} "
              f"--chips {args.chips} --data {args.data} "
              f">> {out / 'logs' / 'supervisor.log'} 2>&1")
        return 0

    # One supervisor per output directory. Two would each restart the other's ranks after every
    # death and the run would never advance, while both logs showed steps.
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "supervisor.pid"
    if lock.exists():
        old = lock.read_text().strip()
        if old.isdigit() and Path(f"/proc/{old}").exists():
            print(f"[sup] supervisor pid {old} already owns {out}; refusing to start a second")
            return 3
    lock.write_text(f"{os.getpid()}\n")

    restarts = 0
    t0 = time.monotonic()
    try:
        while True:
            print(f"[sup] launching {len(chips)} rank(s) on chips {chips} "
                  f"(restart {restarts})", flush=True)
            procs = [launch(r, chips, args) for r in range(len(chips))]
            while True:
                time.sleep(2.0)
                dead = [p for p in procs if p.poll() is not None]
                if not dead:
                    continue
                codes = {p.pid: p.returncode for p in dead}
                if all(p.poll() == 0 for p in procs):
                    print(f"[sup] every rank exited 0 after {time.monotonic() - t0:.0f}s")
                    return 0
                # A rank exited. Whether it finished or died, the world restarts together,
                # because a partially live world is the diverging-replica failure.
                print(f"[sup] rank(s) gone: {codes}", flush=True)
                kill_all(procs, "one rank is gone, the world restarts together")
                break
            done = list(out.glob("COMPLETE-rank*"))
            if len(done) == len(chips):
                print(f"[sup] run complete: {[p.name for p in done]}")
                return 0
            restarts += 1
            if restarts > args.max_restarts:
                print(f"[sup] {restarts} restarts exceeds --max-restarts; stopping so a "
                      f"crash loop cannot burn the card for days")
                return 4
            if args.max_seconds and time.monotonic() - t0 > args.max_seconds:
                print(f"[sup] --max-seconds reached")
                return 0
            # The rendezvous holds one dead world's files. Clearing it before the restart
            # keeps a stale gradient from a killed rank out of the resumed run's first reduce.
            import shutil
            shutil.rmtree(args.rendezvous, ignore_errors=True)
            args.kill_at = 0  # the demo kill fires once, not on every restart
            print(f"[sup] settling {args.settle:.0f}s, then resuming from the last checkpoint",
                  flush=True)
            time.sleep(args.settle)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
