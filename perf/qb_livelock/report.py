#!/usr/bin/env python3
"""Compare the teardown cost of each arm of the sweep.

The number that matters is how long exit_mmap() runs, because that is the work item the kernel
hands to the bounded per-CPU system_wq whenever somebody other than the dying task drops the last
mm reference. Attribution is by pid: a run's own processes are the ones that wrote to its
footprint file, so an exit_mmap event only counts for an arm if its pid is one of them. Anything
else in the window (another worker's parity gate exits on cards 1-3 the whole time) is reported
separately and never mixed in.
"""
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "perf/qb_livelock/out")
ARMS = ["control", "arena", "trim", "both", "control2", "reorder", "control3"]
WATCHDOG_US = 10000     # the kernel's own "hogged CPU for >10000us" threshold

EX = re.compile(r"EXITMMAP \d+ us=(\d+) map_count=(\d+) total_vm_kB=(\d+) comm=(\S+) pid=(\d+)")
FP = re.compile(r"pid=(\d+) tag=(\S+) rss_kB=(\d+) vmas=(\d+)")


def arm(tag):
    d = OUT / tag
    if not (d / "summary.txt").is_file():
        return None
    summary = (d / "summary.txt").read_text()
    fps = {int(m.group(1)): (int(m.group(3)), int(m.group(4)))
           for m in FP.finditer((d / "footprint.txt").read_text()) } if (d / "footprint.txt").is_file() else {}
    mine, other = [], []
    for m in EX.finditer((d / "probe.txt").read_text()):
        us, mc, tv, comm, pid = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
        (mine if pid in fps else other).append((us, mc, tv, comm, pid))
    mine.sort(reverse=True)
    wall = re.search(r"wall_s=([\d.]+)", summary)
    ladder = re.search(r"ladder_reports=(\S+)\s+last_rung=(\S+)", summary)
    cif = re.search(r"cif_md5=(.*)", summary)
    return dict(
        tag=tag,
        wall=float(wall[1]) if wall else float("nan"),
        ladder=ladder[1] if ladder else "?",
        rung=ladder[2] if ladder else "?",
        cif=(cif[1].strip() if cif else ""),
        mine=mine,
        over=[e for e in mine if e[0] >= WATCHDOG_US],
        other_over=[e for e in other if e[0] >= WATCHDOG_US],
        fps=fps,
    )


rows = [r for r in (arm(t) for t in ARMS) if r]
print(f"{'arm':9s} {'wall_s':>7s} {'teardown_ms':>12s} {'max_ms':>7s} {'>10ms':>6s} "
      f"{'peak_vmas':>10s} {'peak_mapped_GB':>15s} {'rss_at_exit_MB':>15s} {'ladder':>8s}")
for r in rows:
    tot = sum(e[0] for e in r["mine"]) / 1000
    mx = max((e[0] for e in r["mine"]), default=0) / 1000
    pv = max((e[1] for e in r["mine"]), default=0)
    pm = max((e[2] for e in r["mine"]), default=0) / 1e6
    rss = max((v[0] for v in r["fps"].values()), default=0) / 1024
    print(f"{r['tag']:9s} {r['wall']:7.1f} {tot:12.1f} {mx:7.1f} {len(r['over']):6d} "
          f"{pv:10d} {pm:15.2f} {rss:15.0f} {r['ladder']:>8s}")

print("\nper-process teardowns, this run's own pids only (us, map_count, mapped_kB, comm, pid):")
for r in rows:
    print(f"  {r['tag']}:")
    for e in r["mine"]:
        rss = r["fps"].get(e[4], ("-",))[0]
        print(f"    {e[0]/1000:8.1f} ms  vmas={e[1]:5d}  mapped={e[2]/1e6:5.2f} GB  "
              f"rss_at_hook={rss if rss=='-' else f'{rss/1024:.0f}'} MB  {e[3]} pid={e[4]}")
    if r["other_over"]:
        print(f"    [not this run: {len(r['other_over'])} other >10ms teardowns in the window, "
              f"max {max(e[0] for e in r['other_over'])/1000:.1f} ms]")

print("\nparity (cif md5 per arm):")
for r in rows:
    print(f"  {r['tag']:9s} {r['cif'] or 'MISSING'}")
same = {r["cif"] for r in rows if r["cif"]}
print(f"  -> {'IDENTICAL across all arms' if len(same) == 1 else f'DIFFERS: {same}'}")
