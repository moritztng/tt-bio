#!/usr/bin/env python3
"""What qb2's watchdog resets cost the 5-day training leg, from the boot table rather than a guess.

Owner: ``train-orchestrator``. The reset CAUSE is not this script's business and must not be
re-derived here: qb2 has a silent stall that the SP5100 TCO watchdog recovers from, root-caused
as far as it has been by ``qbfix-orchestrator`` (2026-09-14/15) and never cured. Signature on
every one of them, re-confirmed 2026-09-19: ``bootstatus=32`` (WDIOF_CARDRESET), empty pstore,
previous boot ending with no shutdown record.

What IS this script's business is the number that decides whether a multi-day run on that box is
viable, and nobody had it: **how often, and what does each one cost the leg.** Run it against the
live box; every figure below is computed from ``journalctl --list-boots``, which is the only
record of a reset that leaves no shutdown behind.

A boot's "uptime" here is its own last journal entry minus its first. The gap to the NEXT boot is
firmware time, and it separates two populations worth keeping apart -- see ``--verbose``.
"""
import argparse
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone

# The leg's grant. Anchored on the first launch and immutable; see tt_bio/train/deadline.py.
DEADLINE = datetime(2026, 9, 24, 1, 52, 58, tzinfo=timezone.utc)
# Measured 2026-09-19 pass 106 on a REAL reset, not estimated: boot 04:48:39, cron fired the
# @reboot hook at 04:48:48, first completed step by 04:51:04. Reset-to-stepping, including the
# hook's deliberate 120 s settle and the ~130 s of stall-plus-firmware before the box came back.
RESTART_SECONDS = 330.0
# The dispatch park interval, which is what a reset costs when nothing restarts the run.
UNHOOKED_SECONDS = 3 * 3600.0
# The cost everyone forgets. A resume replays every step since the last checkpoint, so the
# expected replay per reset is HALF the checkpoint interval -- and at --checkpoint-minutes 30
# that dominates the restart itself by roughly 3 to 1. The 04:46Z reset replayed steps 684-739.
CHECKPOINT_MINUTES = 30.0

ROW = re.compile(r"^\s*(-?\d+)\s+([0-9a-f]{32})\s+(\w{3} \d{4}-\d\d-\d\d \d\d:\d\d:\d\d \w+)\s+"
                 r"(\w{3} \d{4}-\d\d-\d\d \d\d:\d\d:\d\d \w+)\s*$")


def parse_stamp(s: str) -> datetime:
    return datetime.strptime(s, "%a %Y-%m-%d %H:%M:%S %Z").replace(tzinfo=timezone.utc)


def boots(host: str) -> list:
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                          "journalctl --list-boots --no-pager"],
                         capture_output=True, text=True, timeout=120)
    rows = []
    for line in out.stdout.splitlines():
        m = ROW.match(line)
        if m:
            rows.append({"idx": int(m.group(1)), "first": parse_stamp(m.group(3)),
                         "last": parse_stamp(m.group(4))})
    if not rows:
        raise RuntimeError(f"no boot rows parsed; stderr={out.stderr.strip()[:300]}")
    return sorted(rows, key=lambda r: r["idx"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="qb2")
    ap.add_argument("--since", default="2026-09-16T16:19:00",
                    help="ignore boots before this. The default drops the 26-boot storm of "
                         "2026-09-15/16, which was a different regime -- the watchdog was armed "
                         "at 60s against a 150s design value and was itself causing most of the "
                         "resets it was meant to recover from")
    ap.add_argument("--step-seconds", type=float, default=11.282,
                    help="warm median; pass 105 measured this at AICLK 1350 during the run")
    ap.add_argument("--checkpoint-minutes", type=float, default=CHECKPOINT_MINUTES)
    ap.add_argument("--step", type=int, required=True,
                    help="the CURRENT step, read live. There is no safe default: a stale one "
                         "silently shifts every projection below it")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    try:
        rows = boots(a.host)
    except Exception as exc:                                   # noqa: BLE001
        print(f"UNKNOWN: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc)
    kept = [r for r in rows if r["first"] >= since]
    # The CURRENT boot has not ended, so its uptime is a lower bound and must not enter the mean.
    closed = [r for r in kept if r["idx"] != 0]
    if len(closed) < 2:
        print("UNKNOWN: fewer than two closed boots in window", file=sys.stderr)
        return 2

    ups = [(r["last"] - r["first"]).total_seconds() / 3600.0 for r in closed]
    # Firmware time: this boot's last journal line to the NEXT boot's first. Keyed by boot index
    # rather than by list position, because `closed` is a filtered view of `rows` and pairing the
    # two by position silently misaligns the columns.
    by_idx = {r["idx"]: r for r in rows}
    gaps = {}
    for r in closed:
        nxt = by_idx.get(r["idx"] + 1)
        if nxt:
            gaps[r["idx"]] = (nxt["first"] - r["last"]).total_seconds()

    if a.verbose:
        print(f"{'boot':>5} {'uptime h':>9} {'gap to next s':>14}")
        for r, u in zip(closed, ups):
            g = gaps.get(r["idx"])
            print(f"{r['idx']:>5} {u:>9.2f} {('%.0f' % g) if g is not None else '-':>14}")
        vals = sorted(round(g) for g in gaps.values())
        print(f"\nfirmware gaps: {vals}")
        print("Two populations, and the split is clean: a short one and a long one, nothing "
              "between. Both carry bootstatus=32, so both are the watchdog firing; the long one "
              "is the board taking a fuller path back. Recorded, not explained.\n")

    mean_up, med_up = statistics.mean(ups), statistics.median(ups)
    now = datetime.now(timezone.utc)
    left_h = (DEADLINE - now).total_seconds() / 3600.0

    print(f"window since {since:%Y-%m-%d %H:%M}Z: {len(closed)} closed boots on {a.host}")
    print(f"  uptime  mean {mean_up:.2f} h  median {med_up:.2f} h  "
          f"min {min(ups):.2f}  max {max(ups):.2f}")
    print(f"  rate    {24.0 / mean_up:.2f} resets/day")
    print()
    print(f"grant left {left_h:.2f} h, to {DEADLINE:%Y-%m-%dT%H:%M:%S}Z")
    exp = left_h / mean_up
    print(f"  expected resets before the cap: {exp:.1f}")
    replay_h = a.checkpoint_minutes / 2.0 / 60.0        # expected, uniform over the interval
    print(f"  per reset: {RESTART_SECONDS / 60:.1f} min to restart (measured) + "
          f"{replay_h * 60:.1f} min replayed from the last checkpoint "
          f"(half of --checkpoint-minutes {a.checkpoint_minutes:.0f})")
    print()

    def report(label: str, restart_s: float, ck_min: float) -> float:
        lost_h = exp * (restart_s / 3600.0 + ck_min / 2.0 / 60.0)
        steps = (left_h - lost_h) * 3600.0 / a.step_seconds
        print(f"  {label:<34} {lost_h:>5.2f} h lost ({100 * lost_h / left_h:>4.1f} %), "
              f"~{steps:,.0f} more steps")
        return lost_h

    report("hook absent, checkpoint 30 min", UNHOOKED_SECONDS, a.checkpoint_minutes)
    now_h = report("AS RUNNING: hook, checkpoint 30", RESTART_SECONDS, a.checkpoint_minutes)
    for ck in (10.0, 5.0):
        h = report(f"hook, checkpoint {ck:.0f} min", RESTART_SECONDS, ck)
        n = left_h / (ck / 60.0)
        print(f"{'':>38}saves {now_h - h:.2f} h = ~{(now_h - h) * 3600 / a.step_seconds:,.0f} "
              f"steps, for {n:,.0f} extra writes at under 1 s each (~{n / 3600:.2f} h)")

    lost_h = now_h
    steps = (left_h - lost_h) * 3600.0 / a.step_seconds
    print(f"\nas running, at {a.step_seconds:.3f} s/step, the cap lands near step "
          f"{a.step + steps:,.0f} = {100 * (a.step + steps) / 193512:.2f} % of the "
          f"193,512 schedule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
