#!/usr/bin/env python3
"""Is this HOST quiet enough for a timing read? Run it beside pair_idle.py, not instead of it.

pair_idle.py covers the board-power coupling between the two chips of one p300c. It does not cover
the host, and the host is the channel that retracted a real number on 2026-09-18: c14-matmul-ceiling
measured on node 3 with the sibling idle, benchlock held and the clock forced 1350-1350 during every
fold, and still had to withdraw its +0.2510 s. The release gate was folding on node 0, the other
BOARD, for the whole window. Same host, same PSU, same host DRAM and PCIe: base spread came out
4.35 % over n=8 and the base median sat 0.37 s above the number of record, so the A/A floor
(+/-0.3724 s) swallowed the 0.251 s effect.

benchlock cannot be the guard here either. It serialises benchlock CALLERS, and neither the release
gate nor a hand-run fold takes the lock, so a waiter can be handed a lock on a busy box and told
nothing. Three C14 rows walked through that window in one hour the same day.

    exit 0  host quiet, timing read admissible
    exit 1  NOT QUIET, named contention -- do not measure; wait it out or DEFER
    exit 2  usage/environment problem

    python3 host_quiet.py                  # report and exit status
    python3 host_quiet.py --quiet          # exit status only
    python3 host_quiet.py --maxload 2.0    # loadavg ceiling, default 2.0

What counts as contention, in the order the checks fire:
  1. any live `release_gate.py` process, whether or not it currently holds a device. The gate folds
     continuously for hours in ~20 s children, so sampling for a device fd at one instant misses it
     between children.
  2. any OTHER process holding an fd on /dev/tenstorrent/* AND making CPU progress over a 2 s
     sample. The progress filter is what keeps an idle lease-holder or a clock pin (force_aiclk
     sleeps, so it ticks nothing) from blocking a waiter forever -- that stall is the 2026-08-12
     benchlock incident and is not worth re-creating here.
  3. loadavg over --maxload.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Both overridable for the NEGATIVE CONTROL only. A check that cannot be made to say "quiet" is
# indistinguishable from one that is hard-wired to say "busy", and this one has to be trusted to
# release a measurement, not just to block it. Never set either in a real run.
DEV = Path(os.environ.get("HOST_QUIET_DEV_PREFIX", "/dev/tenstorrent"))
GATE_RE = re.compile(os.environ.get("HOST_QUIET_GATE_RE", "release_gate"))
TICK_THRESHOLD = 2          # clock ticks of CPU over the sample window == "making progress"
SAMPLE_S = 2.0


def _mine() -> set[int]:
    """This process and its ancestors. A driver must not flag its own launcher or its ssh shell."""
    out, pid = set(), os.getpid()
    while pid and pid not in out:
        out.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return out


def _ticks(pid: int):
    try:
        f = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return int(f[11]) + int(f[12])          # utime + stime
    except (OSError, IndexError, ValueError):
        return None


def _argv(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except OSError:
        return ""


def _owner(pid: int) -> str:
    """A folds cwd is its worker worktree, which names the ROW. The gates device-holding child is
    a bare `python3 -c from multiprocessing.spawn ...`, so argv identifies nothing and cwd is the
    only actionable identity."""
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return "?"
    parts = cwd.split("/wt/")
    return parts[1].split("/")[0] if len(parts) > 1 else cwd


def device_fd_holders() -> set[int]:
    """PIDs with an fd on a tt device node. Same-user /proc is readable on this fleet, so no sudo."""
    out = set()
    for d in Path("/proc").glob("[0-9]*/fd"):
        try:
            pid = int(d.parent.name)
        except ValueError:
            continue
        try:
            for fd in d.iterdir():
                try:
                    if os.readlink(fd).startswith(str(DEV)):
                        out.add(pid)
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return out


def gate_pids() -> set[int]:
    try:
        p = subprocess.run(["ps", "-eo", "pid,args", "--no-headers"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    out = set()
    for line in p.stdout.splitlines():
        f = line.split(None, 1)
        if len(f) == 2 and GATE_RE.search(f[1]) and "grep" not in f[1]:
            try:
                out.add(int(f[0]))
            except ValueError:
                pass
    return out - _mine()


def check(maxload: float):
    """-> (ok, lines). Every reason is reported, not just the first, so a caller can see whether it
    is waiting on one gate or on a busy box."""
    lines, ok = [], True
    mine = _mine()

    gates = gate_pids()
    if gates:
        ok = False
        for pid in sorted(gates):
            lines.append(f"release_gate pid {pid} live (owner {_owner(pid)}) -- it folds in ~20 s "
                         "children, so an instantaneous device check does not see it")

    cands = device_fd_holders() - mine
    t0 = {pid: _ticks(pid) for pid in cands}
    time.sleep(SAMPLE_S if cands else 0)
    for pid in sorted(cands):
        a, b = t0.get(pid), _ticks(pid)
        if a is None or b is None:
            continue
        if b - a >= TICK_THRESHOLD:
            ok = False
            lines.append(f"device fd holder pid {pid} BUSY ({b - a} ticks/{SAMPLE_S:.0f}s, "
                         f"owner {_owner(pid)}): {_argv(pid)[:90]}")
        else:
            lines.append(f"device fd holder pid {pid} idle, not contention (owner {_owner(pid)})")

    load = os.getloadavg()[0]
    if load > maxload:
        ok = False
        lines.append(f"loadavg1 {load:.2f} over ceiling {maxload:.2f}")
    else:
        lines.append(f"loadavg1 {load:.2f} (ceiling {maxload:.2f})")
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxload", type=float, default=2.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    ok, lines = check(a.maxload)
    if not a.quiet:
        for l in lines:
            print(l)
        print("host quiet -- timing read admissible" if ok else
              "HOST NOT QUIET -- your timing read is contaminated even with an idle sibling and a "
              "forced clock. Wait it out or DEFER. Do not measure and explain it afterwards.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
