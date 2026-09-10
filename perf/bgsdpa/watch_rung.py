"""Run a BoltzGen rung and sample whether the host is still dispatching.

The 16948-atom rung burned 5401 s with no log line after 84 s and was SIGKILLed, so its census
hook never fired and slow-vs-wedged was indistinguishable. This samples every process in the
fold's tree: CPU time from /proc/<pid>/stat fields 14-15, and the wchan the main thread is parked
in. A fold that is dispatching burns CPU at close to wall-clock rate per busy process; a wedged
chip parks every process in futex_wait at 0%.

    python3 perf/bgsdpa/watch_rung.py --target-res 2100 --timeout 1500 --tag r2100_probe
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HZ = os.sysconf("SC_CLK_TCK")


def tree(root):
    """Every live descendant of root, root included."""
    seen, stack = [], [root]
    while stack:
        p = stack.pop()
        if p in seen or not pathlib.Path(f"/proc/{p}").exists():
            continue
        seen.append(p)
        try:
            stack += [int(x) for x in
                      pathlib.Path(f"/proc/{p}/task/{p}/children").read_text().split()]
        except OSError:
            pass
    return seen


def cpu_of(pid):
    try:
        f = pathlib.Path(f"/proc/{pid}/stat").read_text()
        # comm can hold spaces and parens; everything after the last ')' is field 3 onward.
        f = f[f.rindex(")") + 2:].split()
        return (int(f[11]) + int(f[12])) / HZ      # utime + stime, fields 14-15
    except (OSError, ValueError, IndexError):
        return 0.0


def wchan_of(pid):
    try:
        return pathlib.Path(f"/proc/{pid}/wchan").read_text().strip() or "-"
    except OSError:
        return "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-res", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--card", default="2")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--env", action="append", default=[])
    args = ap.parse_args()

    work = ROOT / "perf" / "bgsdpa" / "work"
    cmd = [sys.executable, str(ROOT / "perf" / "bgsdpa" / "rung.py"),
           "--target-res", str(args.target_res), "--timeout", str(args.timeout),
           "--card", args.card, "--tag", args.tag]
    for e in args.env:
        cmd += ["--env", e]
    trace = work / f"watch_{args.tag}.log"
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    t0 = time.time()
    prev, rows = {}, []
    with open(trace, "w") as fh:
        fh.write(f"# {' '.join(cmd)}\n# wall_s  n_proc  cpu_delta_s  busiest_wchan\n")
        while proc.poll() is None:
            time.sleep(args.every)
            pids = tree(proc.pid)
            now = {p: cpu_of(p) for p in pids}
            delta = sum(v - prev.get(p, v) for p, v in now.items())
            # only count pids we saw last tick, so a newly spawned worker's accumulated
            # time is not read as a burst.
            busy = max(pids, key=lambda p: now[p] - prev.get(p, now[p]), default=0)
            row = {"wall_s": round(time.time() - t0, 1), "n_proc": len(pids),
                   "cpu_delta_s": round(delta, 2), "wchan": wchan_of(busy)}
            rows.append(row)
            fh.write(f"{row['wall_s']:9.1f}  {row['n_proc']:3d}  {row['cpu_delta_s']:8.2f}  "
                     f"{row['wchan']}\n")
            fh.flush()
            prev = now
    out = {"tag": args.tag, "target_res": args.target_res, "rc": proc.returncode,
           "wall_s": round(time.time() - t0, 1), "every_s": args.every, "samples": rows}
    (work / f"watch_{args.tag}.json").write_text(json.dumps(out, indent=2))
    live = [r for r in rows if r["cpu_delta_s"] > 0.5 * args.every]
    print(f"{args.tag}: rc={proc.returncode} wall={out['wall_s']}s  "
          f"{len(live)}/{len(rows)} samples had the tree burning >50% of one core")


main()
