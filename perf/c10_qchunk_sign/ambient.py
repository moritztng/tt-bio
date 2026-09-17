"""Host CPU witness for the locked window. Device holders alone missed the git pack that
overlapped run baseline1, so every accepted fold now carries what else ran on the host."""
from __future__ import annotations
import json, os, select, sys, time
from pathlib import Path

TICKS = os.sysconf("SC_CLK_TCK")
INTERVAL = 0.25

def cpu_ticks():
    out = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit(): continue
        try:
            stat = (d / "stat").read_text().rsplit(") ", 1)[1].split()
            out[int(d.name)] = (int(stat[11]) + int(stat[12]), int(stat[3]),
                                (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:120])
        except (OSError, IndexError, ValueError): pass
    return out

def main(path, floor_pct=2.0):
    own_session = os.getsid(0)
    previous, last = cpu_ticks(), time.monotonic_ns()
    with Path(path).open("w") as out:
        while not select.select([sys.stdin], [], [], INTERVAL)[0]:
            now, current = time.monotonic_ns(), cpu_ticks()
            span = (now - last) / 1e9
            busy = [{"pid": pid, "cpu_pct": round((t - previous[pid][0]) / TICKS / span * 100, 1),
                     "own": sid == own_session, "argv": argv}
                    for pid, (t, sid, argv) in current.items()
                    if pid in previous and (t - previous[pid][0]) / TICKS / span * 100 >= floor_pct]
            out.write(json.dumps({"monotonic_ns": now, "utc_ns": time.time_ns(), "span_s": span,
                                  "loadavg": os.getloadavg()[0],
                                  "busy": sorted(busy, key=lambda b: -b["cpu_pct"])}) + "\n")
            out.flush()
            previous, last = current, now

if __name__ == "__main__":
    main(sys.argv[1])
