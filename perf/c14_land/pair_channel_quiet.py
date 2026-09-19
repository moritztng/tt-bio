#!/usr/bin/env python3
"""Admissibility for a timing read with the two contention channels kept SEPARATE.

`host_quiet.py` is the right guard on an otherwise empty box and the wrong one here. It fails on
any busy `/dev/tenstorrent/*` fd holder anywhere on the host and on loadavg over 2.00, and since
2026-09-19 01:52Z qb2 carries `train-i-run`'s ABB3 data-parallel campaign on dev2+dev3 with
`--max-seconds 432000`. Two ranks at 3.0-3.3 cores each put a floor of about 6.5 on a 16-core
box's loadavg and keep two device fds busy for five days, so `host_quiet` cannot return 0 on this
host before roughly 2026-09-24 no matter how quiet the rest of the box gets. A guard that cannot
say yes does not protect a measurement, it deletes it -- `perf/c14_land/mmshort_when_quiet.log`
gave up at 02:39:59Z after 60 minutes for exactly that reason.

So this splits what `host_quiet` merges:

  BOARD-PAIR channel (hard, never relaxed). The two chips of a p300c share a board power budget,
  so a busy sibling moves the seconds even under benchlock. Same rule and same pairing as
  `pair_idle.py`: sibling = card XOR 1.

  HOST channel (measured and reported, not merged into one boolean). A busy device fd holder on
  the OTHER board pair couples only through the host: PSU, host DRAM, PCIe and CPU. That coupling
  shifts the LEVEL of a fold, which is what retracted c14-matmul-ceiling's +0.2510 s on
  2026-09-18, when the neighbour was a release gate folding in 20 s children -- a NON-stationary
  neighbour, which shifts the level DURING a session and so leaks into an interleaved delta too.
  A steady multi-day training campaign is the stationary case: it inflates the paired spread but
  it does not drift across the arms of one session.

What that licenses, stated so it cannot be over-read:

  * the absolute seconds from a session admitted here are NOT a fold time of record and must never
    be quoted as one. The neighbour shifts the level.
  * only the interleaved PAIRED ratio is claimable, and the session's own A/A arm is the arbiter.
    If the A/A floor is not narrower than the measured delta, the answer is "not measurable on
    this host under this neighbour" -- a finding, not a number.

Checks, in the order they fire:
  1. board-pair sibling has no fd                                  hard fail
  2. no RUNNING release_gate process -- one blocked in benchlock's own queue holds no
     device fd and burns no CPU, so argv alone is a false positive that this row spent
     67 lock-holding minutes on                                   hard fail (non-stationary)
  3. every other busy device fd holder is on the OTHER pair          hard fail if on mine
  4. loadavg1 under --maxload at BOTH ends of a --settle window, and moving by no more than
     --drift between them                                           hard fail (stationarity)

    exit 0  admissible for a PAIRED ratio, under the caveats above
    exit 1  not admissible, with every reason named
    exit 2  usage/environment problem

Negative and positive control:
    python3 pair_channel_quiet.py --card 0                      # refuses while the sibling folds
    C14_PCQ_DEV_PREFIX=<dir of empty files 0..3> \
    C14_PCQ_GATE_RE='__no_such_process__' \
    python3 pair_channel_quiet.py --card 0 --maxload 999 --drift 999 --settle 0
                                                                # must return 0
A check that can only say "busy" is indistinguishable from one hard-wired to say "busy", and this
one has to be trusted to RELEASE a measurement. The gate channel carries its own paired control
in perf/c14_land/test_pair_channel_gate_liveness.py: a sleeping gate must be admitted and a
CPU-burning one under the same argv must still be refused.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PAIRS = {0: 1, 1: 0, 2: 3, 3: 2}
DEV = Path(os.environ.get("C14_PCQ_DEV_PREFIX", "/dev/tenstorrent"))
GATE_RE = re.compile(os.environ.get("C14_PCQ_GATE_RE", "release_gate"))
TICK_THRESHOLD = 2
OWNER_RE = re.compile(r"/\.coworker/wt/([A-Za-z0-9._-]+)")


def _mine() -> set[int]:
    out, pid = set(), os.getpid()
    while pid and pid not in out:
        out.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return out


def _argv(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf8", "replace").strip()
    except OSError:
        return ""


def _owner(pid: int) -> str:
    m = OWNER_RE.search(_argv(pid))
    if m:
        return m.group(1)
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return "?"
    m = OWNER_RE.search(cwd)
    return m.group(1) if m else "?"


def _ticks(pid: int):
    try:
        f = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return int(f[11]) + int(f[12])
    except (OSError, IndexError, ValueError):
        return None


def _descendants(root: int) -> set[int]:
    kids = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            ppid = int((entry / "stat").read_text().split(") ", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(entry.name))
    out, stack = {root}, [root]
    while stack:
        for child in kids.get(stack.pop(), []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def _tree_ticks(root: int):
    """utime+stime over the whole tree, plus the root's reaped-child time.

    A gate that folds ~20 s children burns CPU in those children while they live and
    banks it in the root's cutime/cstime when it reaps them, so this moves either way.
    A gate blocked in benchlock's flock queue moves neither.
    """
    total, seen = 0, False
    for pid in _descendants(root):
        try:
            f = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        except (OSError, IndexError, ValueError):
            continue
        seen = True
        total += int(f[11]) + int(f[12])
        if pid == root:
            total += int(f[13]) + int(f[14])
    return total if seen else None


def holders(card: int) -> list[int]:
    node = DEV / str(card)
    if not node.exists():
        print(f"pair_channel_quiet: {node} does not exist", file=sys.stderr)
        sys.exit(2)
    for cmd in (["fuser", str(node)], ["sudo", "-n", "fuser", str(node)]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if p.returncode in (0, 1):
            return [int(x) for x in p.stdout.split()]
    print("pair_channel_quiet: could not run fuser (needs sudo -n on this host)", file=sys.stderr)
    sys.exit(2)


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


def check(card: int, maxload: float, drift: float, settle: float):
    """-> (ok, lines). Every reason is reported, not only the first."""
    lines, ok = [], True
    mine = _mine()
    sib = PAIRS[card]

    theirs = [p for p in holders(sib) if p not in mine]
    if theirs:
        ok = False
        for pid in sorted(theirs):
            lines.append(f"BOARD PAIR: sibling {sib} held by pid {pid} (owner {_owner(pid)}) -- "
                         f"shares card {card}'s board power budget. Hard fail.")
    else:
        lines.append(f"BOARD PAIR: sibling {sib} has no fd -- the power-budget channel is clear")

    # A gate only perturbs a fold if it is RUNNING. One blocked in benchlock's own flock
    # queue -- which is where my holding the lock puts it -- has no device fd and burns no
    # CPU, so matching its argv alone is a false positive. Same instrument benchlock uses
    # for foreign folds: sample the process tree twice and look for a device fd in it.
    gates = gate_pids()
    if gates:
        on_device = set()
        for other in sorted(PAIRS):
            on_device.update(holders(other))
        g0 = {pid: _tree_ticks(pid) for pid in gates}
        time.sleep(2.0)
        for pid in sorted(gates):
            dev = sorted(_descendants(pid) & on_device)
            a, b = g0.get(pid), _tree_ticks(pid)
            burned = None if (a is None or b is None) else b - a
            if dev or (burned is not None and burned > TICK_THRESHOLD):
                ok = False
                why = f"device fd on {dev}" if dev else f"tree burned {burned} ticks in 2 s"
                lines.append(f"HOST: release_gate pid {pid} live (owner {_owner(pid)}, {why}) "
                             "-- folds in ~20 s children on any chip, a NON-stationary "
                             "neighbour. Hard fail.")
            else:
                lines.append(f"HOST: release_gate pid {pid} present but IDLE (owner "
                             f"{_owner(pid)}): no device fd in its tree, {burned} ticks in "
                             "2 s. Blocked in the benchlock queue, not folding -- not a "
                             "blocker.")

    # every other busy device fd holder, classified by pair
    busy = {}
    for c in sorted(PAIRS):
        for pid in holders(c):
            if pid not in mine:
                busy.setdefault(pid, set()).add(c)
    t0 = {pid: _ticks(pid) for pid in busy}
    if busy:
        time.sleep(2.0)
    for pid, cards in sorted(busy.items()):
        a, b = t0.get(pid), _ticks(pid)
        if a is None or b is None:
            continue
        same = sorted(c for c in cards if c in (card, sib))
        other = sorted(c for c in cards if c not in (card, sib))
        if b - a < TICK_THRESHOLD:
            lines.append(f"HOST: pid {pid} holds dev{cards} but is idle (owner {_owner(pid)})")
            continue
        rate = (b - a) / 2.0 / 100.0
        if same:
            ok = False
            lines.append(f"BOARD PAIR: pid {pid} BUSY on dev{same} ({rate:.2f} cores, owner "
                         f"{_owner(pid)}) -- my own pair. Hard fail.")
        else:
            lines.append(f"HOST: pid {pid} BUSY on dev{other} ({rate:.2f} cores, owner "
                         f"{_owner(pid)}) -- OTHER board pair, host channel only. Level shift, "
                         "not drift. Ratio stays claimable, absolute seconds do not.")

    l0 = os.getloadavg()[0]
    if settle > 0:
        time.sleep(settle)
    l1 = os.getloadavg()[0]
    if max(l0, l1) > maxload:
        ok = False
        lines.append(f"HOST: loadavg1 {l0:.2f} -> {l1:.2f} over ceiling {maxload:.2f}")
    elif abs(l1 - l0) > drift:
        ok = False
        lines.append(f"HOST: loadavg1 moved {l0:.2f} -> {l1:.2f}, drift {abs(l1 - l0):.2f} over "
                     f"{drift:.2f} -- the neighbour is not stationary")
    else:
        lines.append(f"HOST: loadavg1 {l0:.2f} -> {l1:.2f} over {settle:.0f}s, under ceiling "
                     f"{maxload:.2f} and stationary within {drift:.2f}")
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, required=True, choices=sorted(PAIRS))
    ap.add_argument("--maxload", type=float, default=8.0,
                    help="loadavg1 ceiling. 8.0 admits the ABB3 campaign's ~6.5 core floor on "
                         "qb2's 16 cores and nothing much else")
    ap.add_argument("--drift", type=float, default=1.5,
                    help="max loadavg1 movement across the settle window (stationarity)")
    ap.add_argument("--settle", type=float, default=60.0,
                    help="seconds between the two loadavg samples")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    ok, lines = check(a.card, a.maxload, a.drift, a.settle)
    if not a.quiet:
        for l in lines:
            print(l)
        print("ADMISSIBLE for a paired ratio -- absolute seconds are NOT a fold time of record, "
              "and the session's own A/A arm is the arbiter" if ok else
              "NOT ADMISSIBLE -- do not measure. Wait it out or DEFER.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
