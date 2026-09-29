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

HERE_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE_REPO))

from tt_bio.train import deadline  # noqa: E402
from tt_bio.train.launcher import host_threads  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_ID_ENV = "ABB3_RUN_ID"


def chip_order(args, restarts: int) -> list:
    """Which chip each rank takes, which is allowed to differ after a restart.

    A ``tt-smi -r`` on p300c resets the whole board pair and node numbers are not guaranteed
    stable across it, so over 5 days the run can plausibly come back with its ranks on
    different nodes than it started on. If the result depends on that, the reproduction is not
    reproducible -- "output that depends on which card ran it" is a standing hard stop. This
    exists so the property can be TESTED on purpose in ten minutes instead of discovered on
    day 3, and it is inert unless ``--chips-after-restart`` is given.
    """
    base = [int(c) for c in args.chips.split(",")]
    if restarts and args.chips_after_restart:
        swapped = [int(c) for c in args.chips_after_restart.split(",")]
        if sorted(swapped) != sorted(base):
            raise SystemExit(f"--chips-after-restart {args.chips_after_restart} is not a "
                             f"reordering of --chips {args.chips}: the grant is the pair, and "
                             f"a restart onto a chip outside it takes a co-tenant's card")
        return swapped
    return base


def launch(rank: int, chips: list, args) -> subprocess.Popen:
    env = dict(os.environ)
    chip = chips[rank]
    # The grant and the pin, on the command that opens the device. An unpinned open brings up
    # every visible chip, so this is not belt-and-braces -- without it one rank takes the box.
    env["TT_VISIBLE_DEVICES"] = str(chip)
    env["TT_BIO_LEASE_CARDS"] = ",".join(str(c) for c in chips)
    # The holder identity comes from the environment or the flag, never from a constant: the
    # fleet dispatcher's running-task check reads this file, and a run launched by one row
    # while the file says another row's name reads as a task nobody is running.
    env.setdefault("TT_BIO_LEASE_HOLDER", args.lease_holder)
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
           "--data", args.data, "--split", args.split,
           "--split-csv", args.split_csv, "--structures", args.structures]
    if args.max_seconds:
        cmd += ["--max-seconds", f"{args.max_seconds:.3f}"]
    # 0 means "divide the host across the ranks", which is the only setting that makes sense
    # for a run that spawns one process per chip: torch sizes its intra-op pool from the
    # machine and cannot see its siblings, so an unset width oversubscribes by the rank count.
    # Measured on qb1, 16 physical cores, four micro-batches a rank: four ranks at torch's
    # default 16 threads spend 913.98 s of a 931.05 s step in `losses` and `host_backward`;
    # the same per-rank work with the cores divided spends 2.50 s.
    threads = args.torch_threads or host_threads(len(chips))
    cmd += ["--torch-threads", str(threads)]
    # Also in the environment, because torch reads OMP_NUM_THREADS at import and the thread
    # pool it builds there is what the first op uses.
    env["OMP_NUM_THREADS"] = str(threads)
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
    ap.add_argument("--chips-after-restart", default="",
                    help="rank->chip order to use from the first restart onward, as a "
                         "REORDERING of --chips. Proves the result does not depend on which "
                         "node ran which rank")
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rendezvous", default="/dev/shm/abb3-dp")
    ap.add_argument("--checkpoint-minutes", type=float, default=30.0)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--split", default="train")
    ap.add_argument("--split-csv",
                    default="/home/ttuser/abb3_src/ABodyBuilder3/data/split.csv")
    ap.add_argument("--structures",
                    default="/home/ttuser/abb3_data/data/structures/structures")
    ap.add_argument("--max-restarts", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="per-process budget; --days is what caps the RUN")
    ap.add_argument("--days", type=float, default=0.0,
                    help="the grant, in wall-clock days, anchored on the first launch and "
                         "immutable afterwards (tt_bio.train.deadline)")
    ap.add_argument("--lease-holder", default=os.environ.get("TT_BIO_LEASE_HOLDER",
                                                             "worker:train-i-run"))
    ap.add_argument("--settle", type=float, default=20.0,
                    help="seconds between a death and the restart, for the card to come back")
    ap.add_argument("--kill-at", type=int, default=0, help="demo: kill one rank at this step")
    ap.add_argument("--kill-rank", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="intra-op threads a rank; 0 divides the host's physical "
                         "cores across the ranks, which is what a one-process-per-chip "
                         "run wants. Torch's own default is the whole box per rank")
    ap.add_argument("--print-reboot-hook", action="store_true")
    args = ap.parse_args()
    # Derived from the output directory rather than random, so a supervisor restarted after a
    # reboot stamps the SAME id and still recognises ranks it did not itself launch.
    args.run_id = hashlib.blake2b(str(Path(args.out).resolve()).encode(),
                                  digest_size=8).hexdigest()

    chips = [int(c) for c in args.chips.split(",")]
    out = Path(args.out)
    if args.print_reboot_hook:
        # Every argument that defines the CONFIGURATION has to be here, not just the ones that
        # name the run. The first version printed --out/--steps/--chips/--data/--lease-holder
        # and nothing else, so a reboot would have resumed at the argument defaults: micro 8
        # rather than the 4 it was launched with, which does not fit in DRAM at 256 tokens, and
        # a different rendezvous path. It would have been a different experiment wearing the
        # same checkpoint.
        print(f"@reboot cd {REPO} && PYTHONPATH={REPO} {sys.executable} "
              f"{HERE / 'supervise.py'} --out {out} --steps {args.steps} "
              f"--chips {args.chips} --data {args.data} --split {args.split} "
              f"--micro {args.micro} --global-batch {args.global_batch} "
              f"--tokens {args.tokens} --blocks {args.blocks} --seed {args.seed} "
              f"--rendezvous {args.rendezvous} "
              f"--checkpoint-minutes {args.checkpoint_minutes} "
              + (f"--chips-after-restart {args.chips_after_restart} "
                 if args.chips_after_restart else "") +
              f"--lease-holder {args.lease_holder} "
              + (f"--days {args.days} " if args.days else "") +
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

    grant = None
    if args.days or deadline.read(out) is not None:
        grant = deadline.resolve(out, args.days or None)
        left = deadline.remaining(out)
        print(f"[sup] grant {grant['days']} days, ends "
              f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(grant['deadline']))}, "
              f"{left / 3600.0:.2f} h left", flush=True)
        if left <= 0:
            print("[sup] the grant is spent; not launching. Extending it is Moritz's call and "
                  "is taken by removing deadline.json on purpose, not by relaunching.")
            return 0

    restarts = 0
    t0 = time.monotonic()
    try:
        while True:
            if grant is not None:
                args.max_seconds = deadline.remaining(out)
                if args.max_seconds <= 0:
                    print(f"[sup] grant reached after {restarts} restart(s); stopping")
                    return 0
            here = chip_order(args, restarts)
            if here != chips:
                print(f"[sup] rank->chip order changed to {here} for this restart", flush=True)
            print(f"[sup] launching {len(here)} rank(s) on chips {here} "
                  f"(restart {restarts})", flush=True)
            procs = [launch(r, here, args) for r in range(len(here))]
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
            if len(done) == len(here):
                print(f"[sup] run complete: {[p.name for p in done]}")
                return 0
            restarts += 1
            if restarts > args.max_restarts:
                print(f"[sup] {restarts} restarts exceeds --max-restarts; stopping so a "
                      f"crash loop cannot burn the card for days")
                return 4
            if grant is None and args.max_seconds and time.monotonic() - t0 > args.max_seconds:
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
