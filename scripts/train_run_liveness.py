#!/usr/bin/env python3
"""Is the 5-day leg alive right now, and is it holding the pair it was granted?

Owner: ``train-orchestrator``. This is deliberately NOT a second copy of
``scripts/abb3_port/heartbeat.py`` (``train-i-run``'s, and the instrument of record for the
curve, the schedule and the resume ledger). It answers the two questions that one does not,
and it answers them when that one cannot.

**Why a second instrument at all.** On 2026-09-19 qb2 rebooted uncleanly at 04:29Z and killed
the leg. The heartbeat would have caught that -- and then it latched: the resumed run's first
logged step was 685 while the checkpoint was 683, its ``(after - 1) in checkpoints`` predicate
missed by one, and it has reported ``INCIDENT: a restart did not resume`` ever since. The record
is append-only, so that verdict is stuck for the rest of the leg. A single detector that can go
dark is not supervision, and the leg still had 4.9 days to run.

So this one is built to the opposite constraints:

* **stdlib only, no repo imports.** The heartbeat died on ``ModuleNotFoundError: numpy`` under
  the system interpreter before it said anything about the run. An instrument that needs the
  venv is an instrument that can fail for reasons that have nothing to do with its subject.
* **nothing latched.** Every answer comes from the process table and the newest history row, so
  the verdict describes now and cannot be poisoned by something that happened on Tuesday.
* **instrument failure is exit 2, never exit 1.** Today's crash exited 1, the same code the
  documented reading ("exit 1 means act") reserves for a real incident. Two causes sharing one
  code is how a reader learns to ignore the code.

Exit 0 healthy, 1 the run is down or stalled, 2 this script could not tell.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HOST_DEFAULT = "qb2"
RUN_DEFAULT = "/home/ttuser/.coworker/wt/train-i-run/runs/base-loss"

# One shell round trip, because a probe that opens six ssh connections is slower than the thing
# it is probing and gets skipped on a busy pass.
PROBE = r'''
RUN=%s
if [ -f "$RUN/supervisor.pid" ]; then
  SUP=$(cat "$RUN/supervisor.pid" 2>/dev/null)
  echo "SUPPID $SUP"
  [ -d "/proc/$SUP" ] && echo "SUPALIVE 1" || echo "SUPALIVE 0"
fi
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  case "$pid" in *[!0-9]*) continue;; esac
  cl=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
  case "$cl" in *repro.py*) ;; *) continue;; esac
  # Bind to THIS run's output directory. Any other leg on the box runs the same script, and a
  # probe that counts a neighbour's ranks reports a healthy pair that is not ours. The ranks
  # pass --out RELATIVE ("runs/base-loss"), so it only means anything resolved against their
  # own cwd -- matching the literal string finds nothing, which this probe did on first run.
  o=$(tr '\0' '\n' < "$p/cmdline" 2>/dev/null | grep -A1 -x -- "--out" | tail -1)
  case "$o" in /*) abs="$o";; *) abs="$(readlink "$p/cwd" 2>/dev/null)/$o";; esac
  [ "$abs" = "$RUN" ] || continue
  rank=$(tr '\0' '\n' < "$p/cmdline" 2>/dev/null | grep -A1 -x -- "--rank" | tail -1)
  # [0-9][0-9]* and not [0-9]*: the starred form matches ZERO digits, so an `ls` that races the
  # fd table yields a phantom "tenstorrent/" node and the probe reports DOWN on a healthy run.
  # Seen 2026-09-19 on this script's own negative control.
  nodes=$(ls -l "$p/fd" 2>/dev/null | grep -o "tenstorrent/[0-9][0-9]*" | sort -u | tr '\n' ',')
  [ -n "$nodes" ] || continue
  echo "RANK $pid ${rank:-?} ${nodes%%,}"
done
for f in "$RUN"/history-rank*.jsonl; do
  [ -f "$f" ] || continue
  echo "HIST $(basename "$f") $(wc -l < "$f") $(tail -1 "$f")"
done
# LAST, not first. Sampled before the history read, this clock is older than the newest row a
# step that landed mid-probe wrote, the age comes out negative, and the stall bar can never
# fire. Caught 2026-09-19 by this script's own --stall-seconds 1 control returning LIVE.
echo "NOW $(date -u +%%s)"
''' 


def probe(host: str, run: str) -> str:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
           PROBE % run] if host else ["/bin/sh", "-c", PROBE % run]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0 and not out.stdout.strip():
        raise RuntimeError(f"probe failed rc={out.returncode}: {out.stderr.strip()[:400]}")
    return out.stdout


def parse(raw: str) -> dict:
    st = {"now": None, "sup_pid": None, "sup_alive": None, "ranks": [], "hist": []}
    for line in raw.splitlines():
        f = line.split(None, 3)
        if not f:
            continue
        if f[0] == "NOW":
            st["now"] = int(f[1])
        elif f[0] == "SUPPID":
            st["sup_pid"] = f[1]
        elif f[0] == "SUPALIVE":
            st["sup_alive"] = f[1] == "1"
        elif f[0] == "RANK" and len(f) >= 4:
            st["ranks"].append({"pid": f[1], "rank": f[2], "nodes": f[3].split(",")})
        elif f[0] == "HIST" and len(f) >= 4:
            try:
                row = json.loads(f[3])
            except ValueError:
                row = {}
            st["hist"].append({"file": f[1], "rows": int(f[2]), "last": row})
    return st


def verdict(st: dict, world: int, stall: float, pair: list) -> tuple:
    """(verdict, reasons). Every branch names the evidence it read."""
    bad = []
    if st["now"] is None:
        raise RuntimeError("probe returned no clock line; refusing to guess")

    if st["sup_alive"] is False:
        bad.append(f"supervisor pid {st['sup_pid']} is gone")
    if len(st["ranks"]) != world:
        bad.append(f"{len(st['ranks'])} rank(s) holding a device, expected {world}")

    nodes = sorted(n for r in st["ranks"] for n in r["nodes"])
    # Distinct nodes, because two ranks on one chip still produce a respectable step rate.
    if len(nodes) != len(set(nodes)):
        bad.append(f"ranks share a device node: {nodes}")
    if pair and set(nodes) != {f"tenstorrent/{c}" for c in pair}:
        bad.append(f"nodes {nodes} are not the granted pair {pair}")

    if not st["hist"]:
        bad.append("no history file")
    else:
        newest = max((h["last"].get("t", 0) for h in st["hist"]), default=0)
        # max(0, ...): the probe's clock comes from `date +%s`, which truncates to whole
        # seconds, while a history row's `t` carries fractions -- so a row written in the same
        # second the clock is read can be up to ~1 s "in the future" and print a negative age.
        # Seen 2026-09-19 as "-0.4s ago" on a healthy run. Clamping is honest here because the
        # quantity is staleness, which has no negative values; the stall bar is unaffected.
        age = max(0.0, st["now"] - newest) if newest else None
        st["age"] = age
        st["step"] = max((h["last"].get("step", 0) for h in st["hist"]), default=None)
        if age is None:
            bad.append("newest history row carries no timestamp")
        elif age > stall:
            bad.append(f"newest step is {age:.0f}s old, over the {stall:.0f}s stall bar")
    return ("DOWN" if bad else "LIVE"), bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=HOST_DEFAULT, help="'' to probe locally")
    ap.add_argument("--run", default=RUN_DEFAULT)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--pair", default="2,3", help="granted device nodes, '' to skip the check")
    ap.add_argument("--stall-seconds", type=float, default=600.0,
                    help="a step takes ~11s and a checkpoint pauses it; 600 is ~50 steps")
    a = ap.parse_args()

    try:
        st = parse(probe(a.host, a.run))
        pair = [int(c) for c in a.pair.split(",")] if a.pair else []
        v, bad = verdict(st, a.world, a.stall_seconds, pair)
    except Exception as exc:                       # noqa: BLE001 -- see exit 2 above
        print(f"LIVENESS: UNKNOWN  {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  exit 2 is this script failing, NOT the run. Do not read it as an incident.",
              file=sys.stderr)
        return 2

    print(f"LIVENESS: {v}  {a.run} on {a.host or 'localhost'}")
    for b in bad:
        print(f"  DOWN: {b}")
    print(f"  supervisor {st['sup_pid']} alive={st['sup_alive']}")
    for r in sorted(st["ranks"], key=lambda r: r["rank"]):
        print(f"  rank {r['rank']} pid {r['pid']} on {','.join(r['nodes'])}")
    if st.get("step") is not None:
        print(f"  step {st['step']}, {st.get('age'):.1f}s ago")
    for h in st["hist"]:
        print(f"  {h['file']}: {h['rows']} rows, last step {h['last'].get('step')}")
    print(f"  read at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(st['now']))}")
    return 0 if v == "LIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
