#!/usr/bin/env python3
"""Prove the reproduction's auto-resume by killing it, on the card, at the real configuration.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:train-b3-train \
    PYTHONPATH=$PWD python3 scripts/abb3_port/resume_gate.py --steps 8 --kill-at 4

Two arms of the same run, and the comparison between them is the gate:

* **control** -- ``--steps N`` uninterrupted, recording the master-weight digest after every step;
* **killed** -- the same run with a SIGKILL after step K, restarted by the supervisor, resuming
  from the last checkpoint and carrying on to N.

**The pass condition is bit-identical digests from step K onward, not a plausible loss curve.**
A resume that reloads the weights but leaves the Adam moments at a fresh optimizer's zeros, or
resets the step count driving bias correction and the LR schedule, produces a loss curve that
keeps falling and looks entirely healthy -- it is just optimising a different problem. The
digest is over the float32 masters, so it moves on any of those omissions.

SIGKILL rather than an exception or a signal handler, because that is what the failure mode
actually is: qb2's watchdog resets give the process nothing. A resume proven against a clean
shutdown has not been proven against the event it exists for.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digests(out: Path, rank: int = 0) -> dict:
    """step -> master digest, read from the rank's appended history.

    Last write wins per step, which is the correct reading: a step redone after a resume is the
    step that counts, and the killed arm redoes the step whose checkpoint it lost.
    """
    d, walls, losses = {}, {}, {}
    path = out / f"history-rank{rank}.jsonl"
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        d[row["step"]] = row["digest"]
        walls[row["step"]] = row["wall"]
        losses[row["step"]] = row.get("loss")
    return {"digest": d, "wall": walls, "loss": losses}


def supervise(out: Path, args, *, kill_at: int = 0, extra=()) -> int:
    cmd = [sys.executable, str(HERE / "supervise.py"), "--out", str(out),
           "--steps", str(args.steps), "--chips", args.chips,
           "--micro", str(args.micro), "--tokens", str(args.tokens),
           "--blocks", str(args.blocks), "--seed", str(args.seed),
           "--checkpoint-minutes", str(args.checkpoint_minutes),
           "--settle", str(args.settle), "--max-restarts", "3",
           "--rendezvous", args.rendezvous]
    if kill_at:
        cmd += ["--kill-at", str(kill_at)]
    cmd += list(extra)
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    print(f"[gate] supervisor exited {rc} after {time.monotonic() - t0:.0f}s", flush=True)
    return rc


def _nodes(out: Path, rank: int) -> list:
    """The device nodes rank ``rank`` held, from the provenance it wrote when it finished."""
    path = out / f"provenance-rank{rank}.json"
    if not path.is_file():
        return []
    return list(json.loads(path.read_text()).get("device_nodes") or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/resume-gate")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--kill-at", type=int, default=4)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chips", default="0")
    ap.add_argument("--chips-after-restart", default="",
                    help="resume the killed arm with this rank->chip order, to prove the "
                         "masters do not depend on which node ran which rank")
    ap.add_argument("--rendezvous", default="/dev/shm/abb3-resume-gate")
    #: Every step, so a short gate still exercises a real checkpoint-and-resume. The run itself
    #: uses 30 minutes; what the cadence changes is how much work a reset costs, not whether
    #: the resume works, so shortening it here tests the same mechanism.
    ap.add_argument("--checkpoint-minutes", type=float, default=0.0001)
    ap.add_argument("--settle", type=float, default=5.0)
    args = ap.parse_args()

    root = Path(args.out)
    control, killed = root / "control", root / "killed"
    import shutil
    for d in (control, killed):
        shutil.rmtree(d, ignore_errors=True)

    if supervise(control, args) != 0:
        print("FAIL: the control arm did not finish")
        return 1
    extra = (["--chips-after-restart", args.chips_after_restart]
             if args.chips_after_restart else [])
    if supervise(killed, args, extra=extra, kill_at=args.kill_at) != 0:
        print("FAIL: the killed arm did not finish after its restart")
        return 1

    a, b = digests(control), digests(killed)
    if not a or not b:
        print(f"FAIL: no history written (control {bool(a)}, killed {bool(b)})")
        return 1
    steps = sorted(set(a["digest"]) & set(b["digest"]))
    print(f"\n{'step':>5} {'control digest':<20} {'killed digest':<20} {'match':<6} "
          f"{'control loss':>13} {'killed loss':>12} {'wall s':>8}")
    bad = []
    for s in steps:
        same = a["digest"][s] == b["digest"][s]
        if not same:
            bad.append(s)
        print(f"{s:>5} {a['digest'][s][:16]:<20} {b['digest'][s][:16]:<20} "
              f"{'yes' if same else 'NO':<6} {a['loss'][s]:>13.6f} {b['loss'][s]:>12.6f} "
              f"{b['wall'][s]:>8.2f}")
    # The node each rank ACTUALLY held, read off its own /proc/<pid>/fd during the run and
    # carried in the provenance. Not the chip id we asked for: TT_VISIBLE_DEVICES=3 has opened
    # /dev/tenstorrent/0 on this host, so the request and the node are different facts.
    before = {r: _nodes(control, r) for r in range(len(args.chips.split(",")))}
    after = {r: _nodes(killed, r) for r in before}
    moved = [r for r in before if before[r] and after[r] and before[r] != after[r]]
    print(f"\nNODES control {before} -> killed {after}")
    if args.chips_after_restart:
        if not moved:
            print(f"FAIL: --chips-after-restart {args.chips_after_restart} was asked for and no "
                  f"rank changed node, so the cross-chip property was NOT exercised. A pass "
                  f"here would be a pass on the same-chip gate wearing a different name")
            return 1
        print(f"CROSS-CHIP: ranks {moved} resumed on a node they did not start on")

    restarts = len(list((killed / 'logs').glob('rank0.log')))
    resumed = [ln for ln in (killed / "logs" / "rank0.log").read_text().splitlines()
               if "resumed from" in ln]
    print(f"\nRESUMES PERFORMED: {len(resumed)} -- {resumed}")
    if not resumed:
        print("FAIL: the killed arm never resumed from a checkpoint, so nothing was proven")
        return 1
    if bad:
        print(f"FAIL: digests differ at steps {bad}. The resume restored some state and not "
              f"all of it -- masters, both RAdam moments, RAdam's step counter and the dropout "
              f"generator are the complete set")
        return 1
    if max(steps) < args.steps:
        print(f"FAIL: the arms only overlap to step {max(steps)} of {args.steps}")
        return 1
    print(f"\nPASS: bit-identical masters on all {len(steps)} shared steps, across a SIGKILL "
          f"after step {args.kill_at} and an unattended resume. The killed arm lost step "
          f"{args.kill_at} and redid it, which is what a reset costs.")
    walls = sorted(a["wall"].values())
    print(f"STEP (control arm, 1 chip): median {walls[len(walls) // 2]:.3f} s over "
          f"{len(walls)} steps, min {walls[0]:.3f}, max {walls[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
