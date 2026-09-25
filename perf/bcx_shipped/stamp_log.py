#!/usr/bin/env python3
"""Stamp a running campaign's stdout with UTC, and die when the campaign does.

BindCraft 2's CSV tables carry no clock (`!_Trajectories.csv` has 40 columns and not one of
them is a time), so the only wall-clock record of a trajectory is its stdout, and an arm
launched without a stamper has none. This reproduces one beside the log.

Two properties, both of them the point:

  * it EXITS when `--pid` is gone, after a final drain. `state/bcx/SHIPPED-NOT-DISPATCHED.md`:
    a leftover `tail -F` from this row outlived the run it followed by three hours, and
    because `fleet.sh`'s `chain_alive` pass 2 matches a helper by its cwd, the row became
    undispatchable for 2.5 h with no log line saying so.
  * it needs no cwd inside a worktree, and should be launched from `/tmp`. A concluded
    slug's worktree is torn down; a helper rooted there either blocks the teardown or loses
    its files mid-run.

Polling rather than `tail -F`: one open/seek/read per interval, no child process, and the
pid check and the drain sit in the same loop so there is exactly one thing to kill.
"""
import argparse
import os
import time


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pid", type=int, required=True, help="the campaign; the stamper exits with it")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    offset, tail = 0, ""
    while True:
        running = alive(args.pid)
        try:
            with open(args.log, errors="replace") as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
        except FileNotFoundError:
            chunk = ""
        if chunk:
            tail += chunk
            lines = tail.split("\n")
            tail = lines.pop()                       # keep a partial line for the next read
            now = time.strftime("%FT%TZ", time.gmtime())
            with open(args.out, "a") as out:
                for line in lines:
                    out.write(f"{now} {line}\n")
        if not running:                              # drained after the subject exited
            with open(args.out, "a") as out:
                out.write(f"{time.strftime('%FT%TZ', time.gmtime())} "
                          f"[stamp_log] pid {args.pid} is gone, stamper exiting\n")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
