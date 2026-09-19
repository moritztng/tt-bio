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
  2. no live release_gate process                                   hard fail (non-stationary)
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
one has to be trusted to RELEASE a measurement.
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

    gates = gate_pids()
    if gates:
        ok = False
        for pid in sorted(gates):
            lines.append(f"HOST: release_gate pid {pid} live (owner {_owner(pid)}) -- folds in "
                         "~20 s children on any chip, a NON-stationary neighbour. Hard fail.")

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
